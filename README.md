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

## Requirements

### Python
- Python 3.14+ (recommended)
- pip or Python package manager

### Ollama
- Ollama installed and running locally
- At least one Ollama model available for analysis (e.g., llama3:8b)
- At least one Ollama model available for embeddings (e.g., qwen3-embedding:4b)

### Dependencies
```
langchain-community
langchain-ollama
python-dotenv
````

## Installation

1. Ensure you have Python 3.14+ installed
2. Install required dependencies:
   ```bash
   pip install langchain-community langchain-ollama python-dotenv
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

### With Environment Variables
Set environment variables in a `.env` file:
```
DEFAULT_MODEL=llama3:8b
EMBEDDING_MODEL=qwen3-embedding:4b
OLLAMA_BASE_URL=http://localhost:11434
```

## How It Works

1. The tool reads the specified log file
2. Splits the log content into manageable chunks for processing
3. Creates a vector store using ChromaDB to store the log content
4. Uses the specified Ollama model for analysis and a separate model for embeddings
5. Identifies errors, warnings, anomalies, and other issues
6. Returns a comprehensive analysis in a structured format

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