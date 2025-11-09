# Edalo - Educational Data LLM Optimizer

A lightweight, local tool that transforms unstructured, multilingual educational data into clean, structured output — easy for any educator to use and any LLM to understand.

## Features

- **Multilingual Support**: Detects and processes content in multiple languages (FR, DE, EN, LU, etc.)
- **Multiple File Formats**: Supports PDF, DOCX, CSV, TXT, and Notion exports
- **Data Optimization**: Cleans duplicates, removes noise, and structures content
- **Embedding Generation**: Creates multilingual embeddings for semantic search
- **Local Processing**: All processing happens locally, no external APIs required
- **LLM Integration**: Works with local LLMs via Ollama

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama (for local LLM):
```bash
# Follow instructions at https://ollama.ai
ollama pull mistral
```

## Usage

### CLI Interface

Process a directory of files:
```bash
python cli.py --input /path/to/files
```

### Python API

```python
from pipeline import run_pipeline

results = run_pipeline("/path/to/files")
```

## Architecture

The pipeline consists of 4 agents:

1. **Format & Language Detector**: Detects file types and languages
2. **Optimizer**: Extracts, cleans, and chunks text
3. **Embedding Generator**: Creates multilingual embeddings
4. **Evaluator**: Measures performance and relevance

## Testing

Generate test data and run tests:
```bash
python tests/generate_test_data.py
python tests/test_pipeline.py
```

## Output

The pipeline generates:
- `output.jsonl`: Structured data in JSONL format
- `output.db`: SQLite database with processed documents
- `chroma_db/`: Vector database with embeddings
- `metrics.json`: Performance metrics and evaluation results

## License

MIT

