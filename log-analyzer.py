# Enhanced version with improved progress tracking
# log-analyzer.py

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import re
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import math

# Suppress annoying Pydantic V1 compatibility warnings for Python 3.14
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

# Modern 2026 LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.retrievers import BaseRetriever

# Load environment variables from .env
load_dotenv()

# Set up logging to output ONLY to run.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run.log", encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Predefined regex patterns to filter out non-critical log lines
CRITICAL_LOG_PATTERNS = [
    r'\bERROR\b',
    r'\bWARN\b',
    r'Exception',
    r'Failed',
    r'exception',
    r'error',
    r'fail',
    r'crash',
    r'fatal',
    r'critical',
    r'timeout',
    r'permission denied',
    r'connection refused',
    r'deadlock',
    r'out of memory',
    r'permission denied',
    r'invalid request',
    r'invalid token',
    r'authentication failed',
    r'access denied',
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze log files using LLM (Python 3.14+)")
    parser.add_argument("log_file", help="Path to the log file to analyze")
    parser.add_argument("--model", default=os.getenv("DEFAULT_MODEL", "llama3:8b"), 
                        help="Ollama model to use for analysis/generation")
    parser.add_argument("--embed-model", default=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b"), 
                        help="Ollama model to use for text embeddings")
    parser.add_argument("--output", default="analysis_output.txt", 
                        help="Output file name")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Force re-indexing of the log file even if embeddings already exist")
    parser.add_argument("--log-dir", default=None,
                        help="Directory containing rotated log files (e.g., syslog.1, syslog.2). "
                             "If provided, will process all log files in the directory and append results.")
    return parser.parse_args()

def is_critical_log_line(line: str) -> bool:
    """Check if a log line contains any critical issue keywords"""
    line = line.strip()
    if not line:
        return False
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in CRITICAL_LOG_PATTERNS)

def filter_chunk(chunk: list[str]) -> list[str]:
    """Helper function to filter a single chunk of lines"""
    return [line for line in chunk if is_critical_log_line(line)]

def filter_critical_logs(log_lines: list[str], num_workers: int = 4) -> list[str]:
    """Filter log lines in parallel by splitting them into chunks"""
    if not log_lines:
        return []

    # Calculate chunk size based on total lines and workers
    chunk_size = math.ceil(len(log_lines) / num_workers)
    chunks = [log_lines[i : i + chunk_size] for i in range(0, len(log_lines), chunk_size)]

    critical_lines = []
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map returns results in the same order as the chunks
        results = executor.map(filter_chunk, chunks)
        
        for result in results:
            critical_lines.extend(result)
            
    return critical_lines

def read_log_file_stream(file_path: str) -> iter:
    """Stream log file line-by-line to avoid loading entire file into memory"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.strip()
    except Exception as e:
        logger.error(f"Error reading log file {file_path}: {str(e)}")
        raise

def get_retriever(log_path: str, embed_model_name: str, force_reindex: bool = False, pbar=None, chunk_count=None):
    """Create or load a persistent vector store and return a retriever with proper progress updates"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    log_filename = Path(log_path).name
    persist_dir = os.path.join(".chroma_db", log_filename)
    
    if pbar: pbar.set_description("Initializing embedding model")
    
    embeddings = OllamaEmbeddings(
        model=embed_model_name,
        base_url=ollama_base_url
    )
    
    if pbar: pbar.update(5)
    
    if os.path.exists(persist_dir) and not force_reindex:
        logger.info(f"Loading existing embeddings from {persist_dir} (Skipping text processing)")
        if pbar: 
            pbar.set_description("Loading cached vector database")
            pbar.update(20)
            
        vector_store = Chroma(
            persist_directory=persist_dir, 
            embedding_function=embeddings
        )
        return vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Process log file and create embeddings in batches
    logger.info(f"Generating new embeddings for {log_path}.")
    
    if pbar: pbar.set_description("Reading log file (streaming)")
    log_lines = list(read_log_file_stream(log_path))
    
    if pbar: pbar.set_description("Filtering critical log lines (Parallel)")
    critical_lines = filter_critical_logs(log_lines, num_workers=os.cpu_count() or 4)
    
    if not critical_lines:
        logger.warning("No critical log lines found. Skipping embedding generation.")
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 5})
    
    if pbar: pbar.set_description("Chunking critical log content")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(" ".join(critical_lines))
    
    if not chunks:
        raise ValueError("No meaningful chunks generated from critical log lines.")
    
    # Use actual chunk count for progress updates
    if chunk_count is None:
        chunk_count = len(chunks)
    
    if pbar: 
        pbar.set_description("Batching embeddings (processing in chunks)")
    
    # Create vector store with batched updates
    vector_store = None
    batch_size = 5
    progress_per_batch = (batch_size / chunk_count) * 20
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        if i == 0:
            vector_store = Chroma.from_texts(
                texts=batch,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        else:
            vector_store.add_texts(texts=batch)
        
        if pbar:
            pbar.update(progress_per_batch)
    
    if pbar:
        pbar.set_description("Embedding process complete")
        pbar.update(10)
    
    return vector_store.as_retriever(search_kwargs={"k": 5})

def analyze_log(log_path: str, model_name: str, embed_model_name: str, force_reindex: bool, pbar=None):
    """Analyze the log using the modern Retrieval Chain (LCEL) with streaming"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if pbar: 
        pbar.set_description("Connecting to LLM")
        pbar.update(5)
    
    llm = OllamaLLM(model=model_name, base_url=ollama_base_url)
    
    retriever = get_retriever(log_path, embed_model_name, force_reindex, pbar)
    
    if pbar: 
        pbar.set_description("Compiling RAG pipeline")
        pbar.update(5)
    
    system_prompt = (
        "You are an expert systems administrator and log analyst. "
        "Use the following pieces of retrieved log context to answer the user's request. "
        "If you don't find the answer in the context, say you don't know.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Use streaming to monitor LLM generation progress
    from langchain_core.runnables import RunnableSequence
    from langchain_core.retrievers import BaseRetriever
    
    try:
        logger.info(f"Invoking RAG chain with generation model: {model_name}")
        if pbar: pbar.set_description("LLM is analyzing retrieved log chunks")
        
        # Enable streaming to update progress as tokens are generated
        chain = prompt | llm
        response = chain.stream({
            "input": "Perform a comprehensive analysis of these logs. Identify any errors, security anomalies, or performance bottlenecks."
        })
        
        # Accumulate response tokens
        full_response = ""
        for chunk in response:
            full_response += chunk.content
            if pbar:
                # Update progress based on chunk generation
                # This provides a more granular view of the LLM processing
                pbar.update(1)
        
        if pbar: pbar.update(30)
        return full_response
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"Error: {str(e)}"

def extract_critical_summary(analysis_result: str, model_name: str = None, pbar=None) -> str:
    """Extract a brief summary of critical issues using the LLM"""
    try:
        if pbar: 
            pbar.set_description("Generating critical summary brief")
            pbar.update(5)
            
        target_model = model_name or os.getenv("DEFAULT_MODEL", "llama3:8b")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        llm = OllamaLLM(model=target_model, base_url=ollama_base_url)
        
        prompt = (
            "Summarize the following log analysis in 50 words or less. "
            "Focus ONLY on the most critical errors, security anomalies, or performance bottlenecks. "
            "If the analysis indicates no major issues, simply state 'No critical issues detected.'\n\n"
            f"Log Analysis:\n{analysis_result}"
        )
        
        logger.info("Generating critical issues summary with LLM...")
        # Enable streaming for summary generation
        summary = ""
        for chunk in llm.stream(prompt):
            summary += chunk.content
            if pbar:
                pbar.update(1)
        
        words = summary.split()
        if len(words) > 100:
            summary = " ".join(words[:100]) + "..."
            
        if pbar: pbar.update(10)
        return summary
        
    except Exception as e:
        logger.warning(f"LLM summarization failed, using text fallback: {str(e)}")
        paragraphs = [p for p in analysis_result.split('\n') if len(p.strip()) > 20]
        if paragraphs:
            fallback_summary = paragraphs[0]
            words = fallback_summary.split()
            if len(words) > 100:
                return " ".join(words[:100]) + "..."
            return fallback_summary
        
        return "Analysis complete. Please review the output file for details."

def send_gotify_notification(file_path: str, status: str, summary: str, pbar=None):
    """Sends a notification to Gotify server with file analysis results"""
    if pbar: pbar.set_description("Sending Gotify push notification")
    
    gotify_server_url = os.getenv("GOTIFY_SERVER_URL", "http://localhost:8080").rstrip('/')
    gotify_token = os.getenv("GOTIFY_TOKEN")
    
    if not gotify_token:
        logger.warning("Gotify token not configured. Notification will not be sent.")
        return
    if not gotify_server_url:
        logger.warning("Gotify server URL not configured. Notification will not be sent.")
        return
    
    payload = {
        "title": f"Log Analysis - {os.path.basename(file_path)}",
        "message": f"Status: {status}\nFile: {file_path}\nSummary: {summary}",
        "priority": 5
    }
    
    try:
        response = requests.post(
            f"{gotify_server_url}/message",
            json=payload,
            headers={
                "X-Gotify-Key": gotify_token,
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully sent Gotify notification for {file_path}")
            if pbar: pbar.update(10)
        else:
            logger.error(f"Failed to send Gotify notification: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error when sending Gotify notification: {str(e)}")

def process_log_directory(log_dir: str, output_file: str, model: str, embed_model: str, force_reindex: bool):
    """Process all log files in a directory, appending results to a single output"""
    if not os.path.exists(log_dir):
        logger.error(f"Log directory {log_dir} does not exist.")
        return
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith(('.log', '.txt'))]
    if not log_files:
        logger.warning("No log files found in directory.")
        return
    
    # Process each log file
    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        logger.info(f"Processing log file: {log_path}")
        
        with tqdm(total=100, desc=f"Processing {log_file}", leave=False) as pbar:
            # Process individual log file
            try:
                analysis_result = analyze_log(
                    log_path=log_path,
                    model_name=model,
                    embed_model_name=embed_model,
                    force_reindex=force_reindex,
                    pbar=pbar
                )
                
                # Save to output file
                with open(output_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(f"\n--- LOG FILE: {log_file} ---\n")
                    out_f.write(str(analysis_result) + "\n")
                
                # Generate summary and send notification
                critical_summary = extract_critical_summary(analysis_result, model, pbar)
                send_gotify_notification(
                    file_path=log_path,
                    status="success" if not str(analysis_result).startswith("Error:") else "failed",
                    summary=critical_summary,
                    pbar=pbar
                )
                
            except Exception as e:
                logger.error(f"Error processing {log_file}: {str(e)}")
                continue

def main():
    try:
        args = parse_arguments()
        
        # If log directory is provided, process all logs in that directory
        if args.log_dir:
            output_file = args.output
            if not output_file.endswith('.txt'):
                output_file += '.txt'
            process_log_directory(
                log_dir=args.log_dir,
                output_file=output_file,
                model=args.model,
                embed_model=args.embed_model,
                force_reindex=args.force_reindex
            )
            return
        
        # Otherwise, process single file
        with tqdm(total=100, desc="Starting Analysis", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]", colour='green') as pbar:
            
            # Phase 1: Retrieve and Analyze
            analysis_result = analyze_log(args.log_file, args.model, args.embed_model, args.force_reindex, pbar)
            
            # Phase 2: Generate Notification Summary
            critical_summary = extract_critical_summary(analysis_result, args.model, pbar)
            
            # Phase 3: File Output
            pbar.set_description("Saving analysis to disk")
            with open(args.output, 'w', encoding='utf-8') as output_file:
                output_file.write(str(analysis_result))
            pbar.update(5)
            
            # Phase 4: Notifications
            if "ENABLE_GOTIFY_NOTIFICATIONS" in os.environ and os.getenv("ENABLE_GOTIFY_NOTIFICATIONS", "false").lower() == "true":
                send_gotify_notification(
                    file_path=args.log_file,
                    status="failed" if str(analysis_result).startswith("Error:") else "success",
                    summary=critical_summary,
                    pbar=pbar
                )
            else:
                pbar.update(10)
            
            # Phase 5: Cleanup & Finish
            pbar.set_description("Done!")
            if pbar.n < 100:
                pbar.update(100 - pbar.n)
            
            pbar.set_description("Done!")
        
        logger.info(f"Analysis complete. Results saved to {args.output}")
        print(f"\n✅ Analysis complete! Check '{args.output}' for the full report and 'run.log' for background execution details.")
            
    except Exception as e:
        logger.error(f"Fatal Error: {str(e)}")
        print(f"\n❌ Fatal Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()