# Edalo Backend Pipeline - Implementation Plan

## Overview
Build a local Python pipeline that processes multilingual educational files (PDF, DOCX, CSV, TXT, Notion exports) through a 4-agent system, outputs optimized structured data (JSONL, SQLite, embeddings), and includes testing framework with metrics.

## Architecture

### Agent 1: Format & Language Detector
- ✅ Detects file types (PDF, DOCX, CSV, TXT, JSON, HTML)
- ✅ Detects languages using langdetect
- ✅ Extracts metadata (file size, creation date, encoding)
- ✅ Output: structured metadata with file type and language labels

### Agent 2: Optimizer
- ✅ Extracts text from various formats (pdfplumber, python-docx, csv.reader, BeautifulSoup)
- ✅ Cleans duplicates using fuzzy matching
- ✅ Removes boilerplate and encoding issues
- ✅ Chunks text into reasonable pieces (sentence/paragraph-based)
- ✅ Output: clean, deduplicated, chunked text with metadata

### Agent 3: Embedding Generator
- ✅ Converts multilingual text to embeddings using sentence-transformers (multilingual-e5-base)
- ✅ Stores embeddings in vector DB (ChromaDB)
- ✅ Output: vector database with indexed chunks

### Agent 4: Evaluator
- ✅ Runs sanity checks on processed data
- ✅ Measures token usage, latency, relevance (via Ollama integration)
- ✅ Compares raw vs optimized data performance
- ✅ Output: metrics report (JSON) and visualization

## Implementation Status

### Phase 1: Core Infrastructure ✅
- [x] **Project Setup**
  - ✅ Created Python project structure
  - ✅ Set up requirements.txt with dependencies
  - ✅ Created config.py for paths and settings
  - ✅ Set up logging infrastructure

- [x] **Data Models** (`models.py`)
  - ✅ FileMetadata dataclass
  - ✅ TextChunk dataclass
  - ✅ ProcessedDocument dataclass
  - ✅ PipelineResults dataclass

### Phase 2: Agents Implementation ✅
- [x] **Agent 1: Format & Language Detector** (`agents/detector.py`)
  - ✅ File type detection function
  - ✅ Language detection function
  - ✅ Metadata extraction function
  - ✅ Main agent orchestrator function
  - ✅ Output formatter function

- [x] **Agent 2: Optimizer** (`agents/optimizer.py`)
  - ✅ Text extraction function (per file type)
  - ✅ Duplicate detection function (fuzzy matching)
  - ✅ Text cleaning function
  - ✅ Text chunking function
  - ✅ Main optimizer orchestrator function

- [x] **Agent 3: Embedding Generator** (`agents/embeddings.py`)
  - ✅ Embedding model loader function
  - ✅ Embedding generation function
  - ✅ Vector DB storage function
  - ✅ Main embedding orchestrator function

- [x] **Agent 4: Evaluator** (`agents/evaluator.py`)
  - ✅ Ollama integration function
  - ✅ Token counting function
  - ✅ Latency measurement function
  - ✅ Relevance scoring function (cosine similarity)
  - ✅ Metrics aggregation function

### Phase 3: Pipeline Orchestration ✅
- [x] **Pipeline Coordinator** (`pipeline.py`)
  - ✅ Main pipeline orchestrator that runs agents sequentially
  - ✅ Error handling and recovery
  - ✅ Progress tracking

### Phase 4: Output & Storage ✅
- [x] **Output Writers** (`output.py`)
  - ✅ JSONL writer function
  - ✅ SQLite writer function
  - ✅ Vector DB writer function (via Agent 3)
  - ✅ Report generator function

### Phase 5: CLI Interface ✅
- [x] **CLI Interface** (`cli.py`)
  - ✅ Command-line argument parsing
  - ✅ File input handling (folder/directory)
  - ✅ Pipeline execution trigger
  - ✅ Results display

### Phase 6: Testing Framework ✅
- [x] **Test Data Generator** (`tests/generate_test_data.py`)
  - ✅ Generate multilingual test files (PDF, DOCX, CSV, TXT)
  - ✅ Create test scenarios (duplicates, noise, multilingual content)
  - ✅ Ground truth data for evaluation

- [x] **Test Runner** (`tests/test_pipeline.py`)
  - ✅ Raw data test (direct LLM query)
  - ✅ Optimized data test (RAG query)
  - ✅ Metrics comparison
  - ✅ Visualization generator

## File Structure
```
EduHack/
├── agents/
│   ├── __init__.py
│   ├── detector.py      # Agent 1
│   ├── optimizer.py     # Agent 2
│   ├── embeddings.py    # Agent 3
│   └── evaluator.py     # Agent 4
├── pipeline.py          # Main orchestrator
├── models.py            # Data models
├── output.py            # Output writers
├── cli.py               # CLI interface
├── config.py            # Configuration
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
├── tests/
│   ├── __init__.py
│   ├── generate_test_data.py
│   ├── test_pipeline.py
│   └── test_data/       # Generated test files
├── input/               # Input directory
├── output/              # Generated outputs
├── README.md
├── PLAN.md
└── .gitignore
```

## Key Constraints Applied
- ✅ Functions max 25 lines
- ✅ Files max 5 functions
- ✅ Use while loops instead of for loops
- ✅ No ternaries
- ✅ Local execution only (no external APIs)

## Dependencies
- langdetect: Language detection
- pdfplumber: PDF text extraction
- python-docx: DOCX parsing
- beautifulsoup4: HTML/Notion export parsing
- sentence-transformers: Multilingual embeddings
- chromadb: Vector database
- ollama: Local LLM integration
- python-magic: File type detection (optional)
- thefuzz: Fuzzy duplicate detection
- reportlab: PDF generation for tests
- matplotlib: Visualization for metrics

## Testing Strategy
1. ✅ Generate representative test data (multilingual student feedback)
2. ✅ Run raw data through Ollama, measure metrics
3. ✅ Run optimized data through pipeline + RAG, measure metrics
4. ✅ Compare: tokens ↓, latency ↓, relevance ↑
5. ✅ Generate metrics report (JSON) and visualization

## Success Criteria
- ✅ Pipeline processes all file types correctly
- ✅ Languages detected accurately
- ✅ Duplicates removed effectively
- ✅ Embeddings generated and stored
- ✅ Metrics show improvement (fewer tokens, faster, better relevance)
- ✅ All tests pass

## Usage

### Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama (for local LLM):
```bash
ollama pull mistral
```

### Generate Test Data
```bash
python tests/generate_test_data.py
```

### Run Tests
```bash
python tests/test_pipeline.py
```

### Use CLI
```bash
python cli.py --input /path/to/files
python cli.py --input /path/to/files --evaluate --query "What are the main issues?"
```

### Python API
```python
from pipeline import run_pipeline

results = run_pipeline(
    input_path=Path("/path/to/files"),
    generate_embeddings_flag=True,
    evaluate=True,
    query="What are the main issues?",
    ground_truth="Expected answer..."
)
```

## Output Files
- `output/output.jsonl`: Structured data in JSONL format
- `output/output.db`: SQLite database with processed documents
- `output/chroma_db/`: Vector database with embeddings
- `output/metrics.json`: Performance metrics and evaluation results
- `output/metrics_visualization.png`: Visualization of improvements
- `output/edalo.log`: Log file

## Next Steps (Future Enhancements)
- [ ] Add web UI with drag-and-drop interface
- [ ] Add support for more file formats (ODT, RTF, etc.)
- [ ] Add batch processing optimization
- [ ] Add API endpoint for programmatic access
- [ ] Add Docker containerization
- [ ] Add CI/CD pipeline
- [ ] Add more comprehensive error handling
- [ ] Add progress bars for long-running operations
- [ ] Add configuration file support (YAML/JSON)
- [ ] Add export to more formats (Parquet, Arrow, etc.)

## Notes
- All processing is done locally
- No external APIs required (except Ollama for LLM)
- Multilingual support for FR, DE, EN, LB, and more
- Configurable chunk sizes and overlap
- Configurable duplicate detection threshold
- Vector database persisted for reuse

## Status
**✅ IMPLEMENTATION COMPLETE**

All planned features have been implemented and tested. The pipeline is ready for use.

