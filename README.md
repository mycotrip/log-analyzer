```markdown
# Log Analyzer - LLM-Powered Log File Analysis Tool

A command-line tool that analyzes log files using LLMs (Large Language Models) for identifying errors, anomalies, and other issues in system logs.

## Overview

This tool leverages the power of modern LLMs (specifically Ollama models) to analyze log files and identify potential issues such as errors, warnings, security anomalies, and performance bottlenecks. The tool uses the LangChain framework with the LCEL (LangChain Expression Language) architecture to create a robust retrieval-augmented generation (RAG) pipeline.

## Features

- Analyzes log files for errors, warnings, and anomalies
- Identifies potential security issues and performance bottlenecks
- Uses modern LangChain LCEL architecture with retrieval-augmented generation
- Configurable Ollama model selection for both analysis and embeddings
- Configurable embedding model selection (default: qwen3-embedding:4b)
- Customizable output file naming
- Comprehensive error handling and logging
- Supports both CLI and script-based usage
- Option to force re-indexing of log files (even if embeddings already exist)
- Separation of analysis model and embedding model for fine-tuned performance
- Support for environment variables to configure model and service settings
- Built-in validation for required environment variables
- Clear documentation for all command-line arguments and options
- Automatic text chunking and splitting using LangChain text splitters
- Vector database integration using Chroma for efficient retrieval
- Dynamic retrieval system that adapts to log content and structure
- Support for multiple log file formats through flexible parsing
- **Gotify notifications** on completion with status, file path, and critical issues summary

## Requirements

### Python
- Python 3.14+ (recommended)
- pip or Python package manager

### Ollama
- Ollama installed and running locally
- At least one Ollama model available for analysis (e.g., llama3:8b)
- At least one Ollama model available for embeddings (e.g., qwen3-embedding:4b)

## Installation

1. Ensure you have Python 3.14+ installed
2. Install required dependencies:
   ```bash
   pip install langchain-community langchain-ollama langchain-chroma langchain-text-splitters python-dotenv requests
   ```

3. Install Ollama (if not already installed):
   - Download from: https://ollama.com/download
   - Follow installation instructions for your operating system

4. Start Ollama service:
   ```bash
   ollama serve
   ```

## Usage

### Basic Usage
```bash
python log-analyzer.py path/to/your/logfile.log
``` 

### With Custom Analysis Model
```bash
python log-analyzer.py path/to/your/logfile.log --model llama3:7b
``` 

### With Custom Embedding Model
```bash
python log-analyzer.py path/to/your/logfile.log --embed-model qwen3-embedding:4b
``` 

### With Custom Output File
```bash
python log-analyzer.py path/to/your/logfile.log --output analysis_results.txt
``` 

### With Force Reindex Option
```bash
python log-analyzer.py path/to/your/logfile.log --force-reindex
``` 
 
## How It Works

1. The tool reads the specified log file
2. Uses LangChain text splitters to divide the log content into manageable chunks for processing
3. Creates a vector store using ChromaDB to store the log content for efficient retrieval
4. Uses the specified Ollama model for analysis and a separate model for embeddings
5. Performs retrieval-based analysis to identify errors, warnings, anomalies, and other issues
6. Returns a comprehensive analysis in a structured format
7. **Sends a Gotify notification** upon completion with status, file path, and critical issues summary

## Output

The tool generates a detailed analysis of the log file, highlighting:
- Errors and warnings
- Security anomalies
- Performance issues
- Suspicious activities
- Potential system issues

The output is saved to a text file by default (named `analysis_output.txt`) or to a custom file if specified.

## Environment Variables

The tool supports several environment variables:

- `DEFAULT_MODEL`: Default Ollama model to use for analysis (default: `llama3:8b`)
- `EMBEDDING_MODEL`: Default Ollama model to use for text embeddings (default: `qwen3-embedding:4b`)
- `OLLAMA_BASE_URL`: Base URL for Ollama service (default: `http://localhost:11434`)
- `ENABLE_GOTIFY_NOTIFICATIONS`: Set to true to enable Gotify notifications (default: false)
- `GOTIFY_SERVER_URL`: Base URL for Gotify server (default: `http://localhost:8080`)
- `GOTIFY_TOKEN`: Authentication token for Gotify (required)
- `GOTIFY_TOPIC`: Topic to send notifications to (default: `logs`)
- `FORCE_REINDEX`: Set to true to force re-indexing of log files even if embeddings already exist (default: false)

## Configuration Options

### Model Selection
- **Analysis Model**: Controls the model used for generating the final analysis output (default: `llama3:8b`)
- **Embedding Model**: Controls the model used for text vectorization and retrieval (default: `qwen3-embedding:4b`)

### Processing Options
- **Force Reindex**: When set to true, forces the tool to re-process the entire log file even if previous embeddings exist. This is useful when you want to apply new analysis rules or update configurations.

### Gotify Integration
- **Enable Notifications**: Set `ENABLE_GOTIFY_NOTIFICATIONS=true` to enable notifications
- **Server URL**: Configure the Gotify server address
- **Authentication**: Provide a valid token for authentication

## Technical Architecture

The tool follows a modular architecture with clear separation of concerns:

1. **Input Processing**: The `load_log_file()` function reads and validates the log file
2. **Text Splitting**: LangChain text splitters divide the log content into manageable chunks
3. **Vector Storage**: ChromaDB stores the text chunks as vectors for efficient retrieval
4. **Retrieval System**: The `get_retriever()` function creates a retriever that can search for relevant log content
5. **Analysis Pipeline**: The analysis model processes the retrieved content to identify issues
6. **Output Generation**: Results are formatted and saved to a file
7. **Notification System**: Sends a Gotify notification with status, file path, and critical issues summary

## Known Limitations

- The tool currently only supports text-based log files
- Performance may degrade with very large log files (over 100MB)
- Some log formats may require additional parsing rules
- Model selection is limited to available Ollama models
- Gotify notifications require a running Gotify server with proper authentication

## License

MIT License

Copyright (c) 2024 Log Analyzer Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.