import argparse
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Modern 2026 LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
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
                        help="Ollama model to use")
    parser.add_argument("--output", default="analysis_output.txt", 
                        help="Output file name")
    return parser.parse_args()

def load_log_file(log_path):
    """Load and read the log file"""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    with open(log_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_retriever(log_content, model_name):
    """Create a vector store and return a retriever"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Split the log content into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(log_content)
    
    # Initialize embeddings with the base_url from .env
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url=ollama_base_url
    )
    
    # Create an in-memory Chroma vector store
    vector_store = Chroma.from_texts(
        chunks, 
        embeddings, 
        metadatas=[{"source": "log_input"} for _ in chunks]
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 5})

def analyze_log(log_content, model_name):
    """Analyze the log using the modern Retrieval Chain (LCEL)"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Use the modern OllamaLLM class (compatible with Pydantic v2/Python 3.14)
    llm = OllamaLLM(model=model_name, base_url=ollama_base_url)
    
    retriever = get_retriever(log_content, model_name)

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

    # Build the modern chain (LCEL)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Execute analysis
    try:
        logger.info(f"Invoking RAG chain with model: {model_name}")
        response = rag_chain.invoke({"input": "Perform a comprehensive analysis of these logs. Identify any errors, security anomalies, or performance bottlenecks."})
        return response["answer"]
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"Error: {str(e)}"

def main():
    try:
        args = parse_arguments()
        log_content = load_log_file(args.log_file)
        
        analysis_result = analyze_log(log_content, args.model)
        
        with open(args.output, 'w', encoding='utf-8') as output_file:
            output_file.write(analysis_result)
        
        logger.info(f"Analysis complete. Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Fatal Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()