# Edalo Architecture Documentation

## Overview

Edalo is a local Python pipeline that processes multilingual educational files through a 4-agent system. The pipeline transforms unstructured data into clean, structured output optimized for LLM consumption.

## System Architecture

### High-Level Flow

```
Input Files → Detector → Optimizer → Embeddings → Evaluator → Output
```

1. **Detector**: Identifies file format and language
2. **Optimizer**: Extracts, cleans, and chunks text
3. **Embeddings**: Generates multilingual embeddings for semantic search
4. **Evaluator**: Measures performance and relevance metrics

## Component Details

### Agent 1: Format & Language Detector (`agents/detector.py`)

**Purpose**: Detect file types and languages from input files.

**Functions**:
- `detect_file_type(file_path)`: Detects file type from extension and MIME type
- `detect_language(text)`: Detects language using langdetect
- `extract_metadata(file_path)`: Extracts file metadata (size, dates, encoding)
- `format_detection_result(...)`: Formats results into FileMetadata object
- `detect_format_and_language(file_path)`: Main orchestrator function

**Input**: File path (Path)
**Output**: FileMetadata object

**Dependencies**:
- `langdetect`: Language detection
- `python-magic`: MIME type detection (optional, falls back to extension)

### Agent 2: Optimizer (`agents/optimizer.py`)

**Purpose**: Extract text, remove duplicates, clean data, and chunk content.

**Functions**:
- `extract_text_pdf(file_path)`: Extract text from PDF files
- `extract_text_docx(file_path)`: Extract text from DOCX files
- `extract_text_csv(file_path)`: Extract text from CSV files
- `extract_text_txt(file_path)`: Extract text from TXT files
- `extract_text_html(file_path)`: Extract text from HTML files
- `extract_text(file_path, file_type)`: Route to appropriate extractor
- `remove_duplicates(text_chunks)`: Remove duplicate chunks using fuzzy matching
- `clean_text(text)`: Clean text (remove noise, fix encoding)
- `chunk_text(text, chunk_size, overlap)`: Chunk text into smaller pieces
- `optimize_data(file_path, metadata)`: Main orchestrator function

**Input**: File path (Path), FileMetadata
**Output**: ProcessedDocument object

**Dependencies**:
- `pdfplumber`: PDF text extraction
- `python-docx`: DOCX parsing
- `beautifulsoup4`: HTML parsing
- `thefuzz`: Fuzzy duplicate detection

### Agent 3: Embedding Generator (`agents/embeddings.py`)

**Purpose**: Generate multilingual embeddings and store in vector database.

**Functions**:
- `load_embedding_model()`: Load sentence-transformers model
- `generate_embeddings(documents)`: Generate embeddings for all documents
- `store_embeddings(chunks, embeddings)`: Store embeddings in ChromaDB
- `create_vector_db(documents)`: Create vector database from documents
- `generate_embeddings(documents)`: Main orchestrator function

**Input**: List of ProcessedDocument objects
**Output**: Boolean (success/failure), vector database path

**Dependencies**:
- `sentence-transformers`: Multilingual embeddings (multilingual-e5-base)
- `chromadb`: Vector database

### Agent 4: Evaluator (`agents/evaluator.py`)

**Purpose**: Evaluate pipeline performance and relevance.

**Functions**:
- `query_ollama(prompt, model)`: Query Ollama LLM
- `count_tokens(text)`: Count tokens in text
- `measure_latency(func)`: Measure function execution time
- `calculate_relevance(query, results)`: Calculate relevance using cosine similarity
- `evaluate_pipeline(documents, query, ground_truth)`: Main orchestrator function

**Input**: List of ProcessedDocument objects, query string, ground truth
**Output**: Evaluation metrics dictionary

**Dependencies**:
- `ollama`: Local LLM integration
- `sentence-transformers`: For relevance calculations

## Data Models (`models.py`)

### FileMetadata
- `file_path`: Path to file
- `file_type`: Detected file type (pdf, docx, csv, txt, etc.)
- `language`: Detected language code
- `file_size`: File size in bytes
- `encoding`: File encoding
- `created_at`: Creation timestamp
- `modified_at`: Modification timestamp

### TextChunk
- `text`: Chunk text content
- `chunk_id`: Unique chunk identifier
- `document_id`: Parent document identifier
- `language`: Language of chunk
- `start_index`: Start position in original text
- `end_index`: End position in original text
- `metadata`: Additional metadata dictionary

### ProcessedDocument
- `file_path`: Path to original file
- `metadata`: FileMetadata object
- `raw_text`: Original extracted text
- `cleaned_text`: Cleaned text (after deduplication and cleaning)
- `chunks`: List of TextChunk objects
- `duplicates_removed`: Number of duplicate chunks removed

### PipelineResults
- `documents`: List of ProcessedDocument objects
- `total_files`: Total number of files processed
- `processed_files`: Number of successfully processed files
- `failed_files`: Number of failed files
- `total_chunks`: Total number of chunks created
- `embeddings_generated`: Boolean indicating if embeddings were generated
- `vector_db_path`: Path to vector database (if generated)
- `evaluation_metrics`: Evaluation metrics dictionary (if evaluated)

## Pipeline Orchestrator (`pipeline.py`)

### Main Function: `run_pipeline()`

**Parameters**:
- `input_path`: Path to input file or directory
- `generate_embeddings_flag`: Whether to generate embeddings (default: True)
- `evaluate`: Whether to run evaluation (default: False)
- `query`: Query string for evaluation (required if evaluate=True)
- `ground_truth`: Ground truth text for evaluation (optional)

**Process**:
1. Collect all supported files from input path
2. For each file:
   - Detect format and language (Agent 1)
   - Optimize data (Agent 2)
   - Add to results
3. Generate embeddings if requested (Agent 3)
4. Run evaluation if requested (Agent 4)
5. Return PipelineResults

### Helper Functions
- `collect_files(input_path)`: Collect all supported files
- `process_single_file(file_path)`: Process a single file through pipeline

## Output Writers (`output.py`)

### Output Formats

1. **JSONL** (`output.jsonl`): One JSON object per line, containing processed documents
2. **SQLite** (`output.db`): SQLite database with documents and chunks tables
3. **ChromaDB** (`chroma_db/`): Vector database with embeddings for semantic search
4. **Metrics** (`metrics.json`): Evaluation metrics and performance data

### Functions
- `write_jsonl(results)`: Write results to JSONL format
- `write_sqlite(results)`: Write results to SQLite database
- `write_metrics(metrics)`: Write metrics to JSON file
- `write_outputs(results)`: Main orchestrator function

## Configuration (`config.py`)

### Settings

- **Directories**: Input, output, test data directories
- **Embedding Model**: `intfloat/multilingual-e5-base`
- **Chunking**: Chunk size (500), overlap (50)
- **Duplicate Detection**: Fuzzy match threshold (0.85)
- **Vector DB**: ChromaDB path and collection name
- **Ollama**: Base URL and model name
- **Output Files**: Paths for JSONL, SQLite, metrics
- **Logging**: Log level and file path

## CLI Interface (`cli.py`)

### Command-Line Arguments

- `--input`: Input file or directory path (required)
- `--no-embeddings`: Skip embedding generation
- `--evaluate`: Run evaluation after processing
- `--query`: Query string for evaluation
- `--ground-truth`: Ground truth text for evaluation

### Usage

```bash
python cli.py --input /path/to/files
python cli.py --input /path/to/files --no-embeddings
python cli.py --input /path/to/files --evaluate --query "What is the main topic?"
```

## Testing (`tests/`)

### Test Data Generator (`tests/generate_test_data.py`)

Generates multilingual test files in various formats:
- PDF files
- DOCX files
- CSV files
- TXT files

### Test Pipeline (`tests/test_pipeline.py`)

Tests the pipeline with generated test data:
- Processes test files
- Verifies output formats
- Checks metrics

## Constraints

### Code Constraints
- Functions: Maximum 25 lines
- Files: Maximum 5 functions
- Loops: Use while loops, not for loops
- Conditionals: No ternaries (use if/else)
- Execution: Local only, no external APIs

### Architecture Constraints
- 4-agent system (Detector, Optimizer, Embeddings, Evaluator)
- Sequential processing through pipeline
- Local processing only (no cloud APIs)
- Multilingual support required

## Dependencies

### Core Dependencies
- `langdetect`: Language detection
- `pdfplumber`: PDF text extraction
- `python-docx`: DOCX parsing
- `beautifulsoup4`: HTML parsing
- `sentence-transformers`: Multilingual embeddings
- `chromadb`: Vector database
- `thefuzz`: Fuzzy duplicate detection
- `ollama`: Local LLM integration
- `python-magic`: File type detection (optional)

### System Dependencies
- `libmagic`: For python-magic (macOS: `brew install libmagic`, Linux: `apt-get install libmagic1`)
- `Ollama`: Local LLM server (https://ollama.ai)

## File Structure

```
EduHack/
├── agents/
│   ├── __init__.py
│   ├── detector.py      # Agent 1: Format & Language Detector
│   ├── optimizer.py     # Agent 2: Optimizer
│   ├── embeddings.py    # Agent 3: Embedding Generator
│   └── evaluator.py     # Agent 4: Evaluator
├── pipeline.py          # Main orchestrator
├── models.py            # Data models
├── output.py            # Output writers
├── cli.py               # CLI interface
├── config.py            # Configuration
├── utils.py             # Utilities
├── tests/
│   ├── __init__.py
│   ├── generate_test_data.py
│   ├── test_pipeline.py
│   └── test_data/       # Generated test files
├── output/              # Generated outputs
│   ├── output.jsonl
│   ├── output.db
│   ├── chroma_db/
│   └── metrics.json
├── input/               # Input files directory
├── README.md
├── WORKFLOW.md
├── CHANGELOG.md
├── ARCHITECTURE.md
└── .cursorrules
```

## Data Flow

1. **Input**: Files in various formats (PDF, DOCX, CSV, TXT, HTML, JSON)
2. **Detection**: File type and language detection
3. **Extraction**: Text extraction from files
4. **Cleaning**: Remove duplicates, clean text, fix encoding
5. **Chunking**: Split text into manageable chunks
6. **Embedding**: Generate embeddings for semantic search
7. **Storage**: Store in JSONL, SQLite, and ChromaDB
8. **Evaluation**: Measure performance and relevance (optional)
9. **Output**: Structured data and metrics

## Error Handling

### Graceful Degradation
- Missing dependencies: Fall back to alternative methods
- File processing errors: Log error, continue with next file
- Language detection failures: Use "unknown" language
- Encoding issues: Attempt multiple encodings

### Logging
- All errors are logged to `output/edalo.log`
- Log level configurable via `LOG_LEVEL` environment variable
- Logs include timestamps, log levels, and detailed error messages

## Future Improvements

### Potential Enhancements
- Support for more file formats
- Improved duplicate detection algorithms
- Better chunking strategies
- Additional evaluation metrics
- Performance optimizations
- Batch processing improvements

### Maintenance
- Regular dependency updates
- Code refactoring for maintainability
- Documentation updates
- Test coverage improvements

