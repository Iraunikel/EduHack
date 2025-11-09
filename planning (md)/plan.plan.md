# Edalo Backend Pipeline Implementation

## Overview

Build a local Python pipeline that processes multilingual educational files (PDF, DOCX, CSV, TXT, Notion exports) through a 4-agent system, outputs optimized structured data (JSONL, SQLite, embeddings), and includes testing framework with metrics.

## Architecture

### Agent 1: Format & Language Detector

- Detects file types (PDF, DOCX, CSV, TXT, JSON)
- Detects languages using langdetect
- Extracts metadata (file size, creation date, etc.)
- Output: structured metadata with file type and language labels

### Agent 2: Optimizer

- Extracts text from various formats (pdfplumber, python-docx, csv.reader, BeautifulSoup)
- Cleans duplicates using fuzzy matching
- Removes boilerplate and encoding issues
- Chunks text into reasonable pieces (sentence/paragraph-based)
- Output: clean, deduplicated, chunked text with metadata

### Agent 3: Embedding Generator

- Converts multilingual text to embeddings using sentence-transformers (multilingual-e5-base)
- Stores embeddings in vector DB (Chroma for simplicity)
- Output: vector database with indexed chunks

### Agent 4: Evaluator

- Runs sanity checks on processed data
- Measures token usage, latency, relevance (via Ollama integration)
- Compares raw vs optimized data performance
- Output: metrics report (JSON) and visualization

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Project Setup**

   - Create Python project structure
   - Set up requirements.txt with dependencies (langdetect, pdfplumber, python-docx, beautifulsoup4, sentence-transformers, chromadb, ollama)
   - Create config file for paths and settings
   - Set up logging

2. **Agent 1: Format & Language Detector** (`agents/detector.py`)

   - File type detection function
   - Language detection function
   - Metadata extraction function
   - Main agent orchestrator function
   - Output formatter function

3. **Agent 2: Optimizer** (`agents/optimizer.py`)

   - Text extraction function (per file type)
   - Duplicate detection function (fuzzy matching)
   - Text cleaning function
   - Text chunking function
   - Main optimizer orchestrator function

4. **Agent 3: Embedding Generator** (`agents/embeddings.py`)

   - Embedding model loader function
   - Embedding generation function
   - Vector DB storage function
   - Main embedding orchestrator function

5. **Agent 4: Evaluator** (`agents/evaluator.py`)

   - Ollama integration function
   - Token counting function
   - Latency measurement function
   - Relevance scoring function (cosine similarity)
   - Metrics aggregation function

### Phase 2: Pipeline Orchestration

6. **Pipeline Coordinator** (`pipeline.py`)

   - Main pipeline orchestrator that runs agents sequentially
   - Error handling and recovery
   - Progress tracking

7. **Data Models** (`models.py`)

   - Data classes for processed documents, chunks, metadata
   - Output format definitions

### Phase 3: Output & Storage

8. **Output Writers** (`output.py`)

   - JSONL writer function
   - SQLite writer function
   - Vector DB writer function (via Agent 3)
   - Report generator function

### Phase 4: CLI Interface

9. **CLI Interface** (`cli.py`)

   - Command-line argument parsing
   - File input handling (folder/directory)
   - Pipeline execution trigger
   - Results display

### Phase 5: Testing Framework

10. **Test Data Generator** (`tests/generate_test_data.py`)

    - Generate multilingual test files (PDF, DOCX, CSV, TXT)
    - Create test scenarios (duplicates, noise, multilingual content)
    - Ground truth data for evaluation

11. **Test Runner** (`tests/test_pipeline.py`)

    - Raw data test (direct LLM query)
    - Optimized data test (RAG query)
    - Metrics comparison
    - Visualization generator

12. **Test Suite** (`tests/`)

    - Unit tests for each agent
    - Integration tests for pipeline
    - Performance benchmarks

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
├── requirements.txt     # Dependencies
├── tests/
│   ├── __init__.py
│   ├── generate_test_data.py
│   ├── test_pipeline.py
│   └── test_data/       # Generated test files
├── output/              # Generated outputs
└── README.md
```

## Key Constraints

- Functions max 25 lines
- Files max 5 functions
- Use while loops instead of for loops
- No ternaries
- Local execution only (no external APIs)

## Dependencies

- langdetect: Language detection
- pdfplumber: PDF text extraction
- python-docx: DOCX parsing
- beautifulsoup4: HTML/Notion export parsing
- sentence-transformers: Multilingual embeddings
- chromadb: Vector database
- ollama: Local LLM integration
- python-magic / filetype: File type detection
- thefuzz: Fuzzy duplicate detection

## Testing Strategy

1. Generate representative test data (multilingual student feedback)
2. Run raw data through Ollama, measure metrics
3. Run optimized data through pipeline + RAG, measure metrics
4. Compare: tokens ↓, latency ↓, relevance ↑
5. Generate metrics report (JSON) and visualization

## Success Criteria

- Pipeline processes all file types correctly
- Languages detected accurately
- Duplicates removed effectively
- Embeddings generated and stored
- Metrics show improvement (fewer tokens, faster, better relevance)
- All tests pass

### To-dos

- [x] Set up project structure, requirements.txt, config.py, and logging infrastructure
- [x] Implement Agent 1: Format & Language Detector (file type detection, language detection, metadata extraction)
- [x] Implement Agent 2: Optimizer (text extraction, duplicate detection, cleaning, chunking)
- [x] Implement Agent 3: Embedding Generator (multilingual embeddings, vector DB storage)
- [x] Implement Agent 4: Evaluator (Ollama integration, token counting, latency, relevance scoring)
- [x] Create pipeline.py orchestrator to run agents sequentially with error handling
- [x] Create models.py with data classes for documents, chunks, and metadata
- [x] Implement output.py for JSONL, SQLite, and report generation
- [x] Create CLI interface with file input handling and pipeline execution
- [x] Create test data generator for multilingual educational content (PDF, DOCX, CSV, TXT)
- [x] Implement test runner comparing raw vs optimized data with metrics and visualization

