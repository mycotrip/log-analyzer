import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Suppress annoying Pydantic V1 compatibility warnings for Python 3.14
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

# Modern 2026 LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

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
    return parser.parse_args()

def load_log_file(log_path):
    """Load and read the log file"""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    with open(log_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_retriever(log_path, embed_model_name, force_reindex=False, pbar=None):
    """Create or load a persistent vector store and return a retriever"""
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
    else:
        logger.info(f"Generating new embeddings for {log_path}...")
        
        if pbar: pbar.set_description("Reading log file")
        log_content = load_log_file(log_path)
        
        if pbar: pbar.set_description("Chunking text")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(log_content)
        
        if not chunks:
            raise ValueError("The provided log file is empty or contains no readable text.")
            
        if pbar: 
            pbar.set_description("Generating embeddings (this may take a moment)")
            pbar.update(10)
        
        vector_store = Chroma.from_texts(
            texts=chunks, 
            embedding=embeddings, 
            metadatas=[{"source": log_path} for _ in chunks],
            persist_directory=persist_dir
        )
        logger.info(f"Embeddings successfully saved to {persist_dir}")
        if pbar: pbar.update(10)
        
    return vector_store.as_retriever(search_kwargs={"k": 5})

def analyze_log(log_path, model_name, embed_model_name, force_reindex, pbar=None):
    """Analyze the log using the modern Retrieval Chain (LCEL)"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if pbar: pbar.set_description("Connecting to LLM")
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

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    try:
        logger.info(f"Invoking RAG chain with generation model: {model_name}")
        if pbar: pbar.set_description("LLM is analyzing retrieved log chunks")
        
        response = rag_chain.invoke({"input": "Perform a comprehensive analysis of these logs. Identify any errors, security anomalies, or performance bottlenecks."})
        
        if pbar: pbar.update(30)
        return response["answer"]
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
        summary = llm.invoke(prompt).strip()
        
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

def main():
    try:
        args = parse_arguments()
        
        # Initialize the fancy TQDM progress bar
        with tqdm(total=100, desc="Starting Analysis", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]", colour='green') as pbar:
            
            # Phase 1: Retrieve and Analyze (approx 60% of process)
            analysis_result = analyze_log(args.log_file, args.model, args.embed_model, args.force_reindex, pbar)
            
            # Phase 2: Generate Notification Summary (approx 15% of process)
            critical_summary = extract_critical_summary(analysis_result, args.model, pbar)
            
            # Phase 3: File Output (approx 5% of process)
            pbar.set_description("Saving analysis to disk")
            with open(args.output, 'w', encoding='utf-8') as output_file:
                output_file.write(analysis_result)
            pbar.update(5)
            
            # Phase 4: Notifications (approx 10% of process)
            if "ENABLE_GOTIFY_NOTIFICATIONS" in os.environ and os.getenv("ENABLE_GOTIFY_NOTIFICATIONS", "false").lower() == "true":
                send_gotify_notification(
                    file_path=args.log_file,
                    status="failed" if analysis_result.startswith("Error:") else "success",
                    summary=critical_summary,
                    pbar=pbar
                )
            else:
                pbar.update(10)
            
            # Phase 5: Cleanup & Finish
            pbar.set_description("Cleanup and finalize")
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