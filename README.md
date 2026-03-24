# Log Analyzer - LLM-Powered Log File Analysis Tool

A command-line tool that analyzes log files using LLMs (Large Language Models) for identifying errors, anomalies, and other issues in system logs.

## Overview

This tool leverages the power of modern LLMs (specifically Ollama models) to analyze log files and identify potential issues such as errors, warnings, security anomalies, and performance bottlenecks. The tool uses the LangChain framework with the LCEL (LangChain Expression Language) architecture to create a robust retrieval-augmented generation (RAG) pipeline.

## Features

- Analyzes log files for errors, warnings, and anomalies
- Identifies potential security issues and performance bottlenecks
- Uses modern LangChain LCEL architecture with retrieval-augmented generation
- Configurable Ollama model selection
- Customizable output file naming
- Comprehensive error handling and logging
- Supports both CLI and script-based usage

## Requirements

### Python
- Python 3.14+ (recommended)
- pip or Python package manager

### Ollama
- Ollama installed and running locally
- At least one Ollama model available (e.g., llama3:8b)

### Dependencies
```
langchain-community
langchain-ollama
python-dotenv
```

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

### With Custom Model
```bash
python log-analyzer.py path/to/your/logfile.log --model llama3:7b
```

### With Custom Output File
```bash
python log-analyzer.py path/to/your/logfile.log --output analysis_results.txt
```

### With Environment Variables
Set environment variables in a `.env` file:
```
DEFAULT_MODEL=llama3:8b
OLLAMA_BASE_URL=http://localhost:11434
```

## How It Works

1. The tool reads the specified log file
2. Splits the log content into manageable chunks for processing
3. Creates a vector store using ChromaDB to store the log content
4. Uses the specified Ollama model to analyze the logs
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

- `DEFAULT_MODEL`: Default Ollama model to use (default: `llama3:8b`)
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