import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
import requests

# Suppress annoying Pydantic V1 compatibility warnings for Python 3.14
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

# Modern 2026 LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma  # <-- Updated to the new dedicated package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
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

def get_retriever(log_path, embed_model_name, force_reindex=False):
    """Create or load a persistent vector store and return a retriever"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Create a unique directory name for this log file's embeddings
    log_filename = Path(log_path).name
    persist_dir = os.path.join(".chroma_db", log_filename)
    
    # Initialize embeddings model
    embeddings = OllamaEmbeddings(
        model=embed_model_name,
        base_url=ollama_base_url
    )
    
    # Check if we already have a database for this log file
    if os.path.exists(persist_dir) and not force_reindex:
        logger.info(f"Loading existing embeddings from {persist_dir} (Skipping text processing)")
        vector_store = Chroma(
            persist_directory=persist_dir, 
            embedding_function=embeddings
        )
    else:
        logger.info(f"Generating new embeddings for {log_path}...")
        log_content = load_log_file(log_path)
        
        # Split the log content into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(log_content)
        
        if not chunks:
            raise ValueError("The provided log file is empty or contains no readable text.")
        
        # Create and persist the Chroma vector store
        vector_store = Chroma.from_texts(
            texts=chunks, 
            embedding=embeddings, 
            metadatas=[{"source": log_path} for _ in chunks],
            persist_directory=persist_dir
        )
        logger.info(f"Embeddings successfully saved to {persist_dir}")
        
    return vector_store.as_retriever(search_kwargs={"k": 5})

def analyze_log(log_path, model_name, embed_model_name, force_reindex):
    """Analyze the log using the modern Retrieval Chain (LCEL)"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Use the modern OllamaLLM class for generation
    llm = OllamaLLM(model=model_name, base_url=ollama_base_url)
    
    # Pass the path instead of content so the retriever can decide whether to read the file
    retriever = get_retriever(log_path, embed_model_name, force_reindex)

    # Define a modern ChatPromptTemplate
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

    # Build the modern chain (LCEL) using langchain_classic
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Execute analysis
    try:
        logger.info(f"Invoking RAG chain with generation model: {model_name}")
        response = rag_chain.invoke({"input": "Perform a comprehensive analysis of these logs. Identify any errors, security anomalies, or performance bottlenecks."})
        return response["answer"]
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"Error: {str(e)}"

def send_gotify_notification(file_path: str, status: str, summary: str):
    """
    Sends a notification to Gotify server with file analysis results
    
    Args:
        file_path: Path to the analyzed log file
        status: Success or failure status
        summary: Brief summary of critical issues (100 words or less)
    """
    # Get environment variables
    gotify_server_url = os.getenv("GOTIFY_SERVER_URL", "http://localhost:8080")
    gotify_token = os.getenv("GOTIFY_TOKEN")
    gotify_topic = os.getenv("GOTIFY_TOPIC", "logs")
    
    # Validate required fields
    if not gotify_token:
        logger.warning("Gotify token not configured. Notification will not be sent.")
        return
    
    if not gotify_server_url:
        logger.warning("Gotify server URL not configured. Notification will not be sent.")
        return
    
    # Prepare notification payload
    payload = {
        "title": f"Log Analysis - {os.path.basename(file_path)}",
        "message": f"Status: {status}\nFile: {file_path}\nSummary: {summary}",
        "topic": gotify_topic
    }
    
    # Send notification via POST request
    try:
        response = requests.post(
            f"{gotify_server_url}/api/v1/pushmsg",
            json=payload,
            headers={
                "X-Gotify-Token": gotify_token,
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully sent Gotify notification for {file_path}")
        else:
            logger.error(f"Failed to send Gotify notification: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error when sending Gotify notification: {str(e)}")

def main():
    try:
        args = parse_arguments()
        
        # We now pass the file path, not the loaded content
        analysis_result = analyze_log(args.log_file, args.model, args.embed_model, args.force_reindex)
        
        # Extract critical issues summary from analysis result
        critical_summary = extract_critical_summary(analysis_result)
        
        # Save results to output file
        with open(args.output, 'w', encoding='utf-8') as output_file:
            output_file.write(analysis_result)
        
        # Send Gotify notification if enabled
        if "ENABLE_GOTIFY_NOTIFICATIONS" in os.environ and os.getenv("ENABLE_GOTIFY_NOTIFICATIONS", "false").lower() == "true":
            send_gotify_notification(
                file_path=args.log_file,
                status="success" if analysis_result.get("success", True) else "failed",
                summary=critical_summary
            )
        
        logger.info(f"Analysis complete. Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Fatal Error: {str(e)}")
        sys.exit(1)

def extract_critical_summary(analysis_result: str) -> str:
    """
    Extract a brief summary of critical issues from the analysis result
    
    Args:
        analysis_result: Full analysis output from the log file
        
    Returns:
        Brief summary of critical issues (under 100 words)
    """
    # Simple extraction - you can enhance this with NLP if needed
    critical_keywords = ["error", "failed", "security", "anomaly", "malicious", "suspicious", "threat", "attack", "breach"]
    summary = ""
    
    # Look for critical issues in the analysis
    for keyword in critical_keywords:
        if keyword in analysis_result.lower():
            summary += f"Found {keyword} in log analysis\n"
            break
    
    # Add a fallback if no critical issues found
    if not summary:
        summary = "No critical issues detected in the log analysis"
    
    # Ensure summary is under 100 characters
    if len(summary) > 100:
        summary = summary[:100] + "..."
    
    return summary.strip()

if __name__ == "__main__":
    main()