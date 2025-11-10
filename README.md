# Edalo - Educational Data LLM Optimizer

A lightweight, local tool that transforms unstructured, multilingual educational data into clean, structured output — easy for any educator to use and any LLM to understand.

## Features

- **Multilingual Support**: Detects and processes content in multiple languages (FR, DE, EN, LU, etc.)
- **Multiple File Formats**: Supports PDF, DOCX, CSV, TXT, HTML (e.g., unzipped Notion page exports), and JSON (text extraction only)
- **Data Optimization**: Cleans duplicates, removes noise, and structures content
- **Embedding Generation**: Creates multilingual embeddings for semantic search
- **Local Processing**: All processing happens locally, no external APIs required
- **LLM Integration**: Works with local LLMs via Ollama

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install system dependencies for python-magic:
```bash
# On macOS:
brew install libmagic

# On Linux:
sudo apt-get install libmagic1  # Debian/Ubuntu
# or
sudo yum install file-devel      # RHEL/CentOS
```

3. Install Ollama (for local LLM):
```bash
# Follow instructions at https://ollama.ai
ollama pull deepseek-coder:6.7b
# Or use any other model you prefer
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

> **Note:** Language detection is most reliable on PDF, DOCX, CSV, and TXT inputs today. HTML/Notion exports currently fall back to `unknown` until sampling improvements land, and zipped Notion packages must be extracted before ingestion.

## Output

The pipeline generates:
- `output.jsonl`: Structured data in JSONL format
- `output.db`: SQLite database with processed documents
- `chroma_db/`: Vector database with embeddings
- `metrics.json`: Performance metrics and evaluation results

## Known limitations & planned improvements

- **HTML language sampling** – Extend `extract_sample_text` to reuse the HTML parsing path so Notion exports contribute language hints before processing.
- **Chunk overlap support** – Update `chunk_text` to build overlapping windows (e.g., sliding stride of `chunk_size - chunk_overlap`) instead of ignoring the configured overlap.
- **Evaluation flexibility** – Allow the evaluator to inject a mockable client, fall back to a lightweight local model, or skip gracefully when Ollama is unavailable, keeping the pipeline runnable in CI.
- **Scalable duplicate detection** – Replace pairwise fuzzy matching with a locality-sensitive hashing or MinHash-based approach to keep deduplication near-linear as corpora grow.
- **Automated end-to-end tests** – Add a pytest scenario that exercises the demo dataset, asserts JSON/HTML ingestion, validates metrics generation, and runs without requiring a live Ollama service by stubbing responses.

## License

MIT

