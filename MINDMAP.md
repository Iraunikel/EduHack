# Edalo - Complete Program Mindmap

## 🎯 Edalo - Educational Data LLM Optimizer
### Purpose: Transform unstructured, multilingual educational data into clean, structured output

### Goals
- Accurately identify file formats and languages in diverse educational data
- Clean, deduplicate, and chunk data for downstream LLM applications
- Generate multilingual text embeddings for improved semantic search and analysis
- Evaluate pipeline output for quality, efficiency, and relevance

### KPIs
- >95% correct detection of file type and language (validation set)
- ≥90% reduction in duplicate or unstructured content
- <10% total processing failure rate
- ≥0.60 average relevance score in output (evaluation metric)
- >2x reduction in LLM token usage for comparable tasks

---

## 📂 Project Structure

**Implementation type**: All modules in this section are **hard coded by us** (project-specific orchestration, models, config, and utilities).

### Core Modules
- **pipeline.py** - Main orchestrator
- **cli.py** - Command-line interface
- **models.py** - Data models
- **output.py** - Output writers
- **config.py** - Configuration settings
- **utils.py** - Utility functions

### Agents Directory (`agents/`)
- **detector.py** - Agent 1: Format & Language Detector
- **optimizer.py** - Agent 2: Data Optimizer
- **embeddings.py** - Agent 3: Embedding Generator
- **evaluator.py** - Agent 4: Evaluator

### Testing (`tests/`)
- **test_pipeline.py** - Pipeline tests
- **test_luxembourgish_detection.py** - Language detection tests
- **generate_test_data.py** - Test data generator
- **test_data/** - Generated test files

### Directories
- **input/** - Input files directory
- **output/** - Generated outputs directory
  - output.jsonl
  - output.db
  - chroma_db/
  - metrics.json
  - edalo.log

---

## 🔄 Pipeline Flow

```
Input Files → Detector → Optimizer → Embeddings → Evaluator → Output
```

### Stage 1: Input Collection
- Collect files from input path (file or directory)
- Supported formats: `.pdf`, `.docx`, `.csv`, `.txt`, `.html`, `.json`
- Recursive directory scanning

### Stage 2: Detection (Agent 1)
- File type detection
- Language detection
- Metadata extraction
- Output: `FileMetadata` object

### Stage 3: Optimization (Agent 2)
- Text extraction
- Text cleaning
- Duplicate removal
- Text chunking
- Output: `ProcessedDocument` object

### Stage 4: Embeddings (Agent 3) - Optional
- Load embedding model
- Generate embeddings for chunks
- Store in ChromaDB
- Output: Vector database

### Stage 5: Evaluation (Agent 4) - Optional
- Query Ollama LLM
- Compare raw vs optimized data
- Calculate metrics
- Output: Evaluation metrics

### Stage 6: Output
- Write JSONL format
- Write SQLite database
- Write metrics JSON
- Generate visualization

---

## 🤖 Agent 1: Format & Language Detector (`agents/detector.py`)

**Implementation type**: Detection flow and metadata logic are **hard coded by us**; detection itself relies on **open source libraries**.

**Local/privacy fit**: Runs fully **local** after Polyglot models are downloaded; **no data is sent outside** the machine.

### Functions (5 functions max)
1. **detect_file_type(file_path)**
   - Detects file type from extension
   - Fallback to MIME type detection (python-magic)
   - Returns: "pdf", "docx", "csv", "txt", "json", "html", "unknown"

2. **detect_language(text)**
   - Uses Polyglot for language detection
   - Supports 130+ languages including Luxembourgish
   - Returns: ISO language code or "unknown"

3. **extract_metadata(file_path)**
   - Extracts file size, encoding, timestamps
   - Detects encoding (UTF-8, UTF-16, etc.)
   - Returns: Dictionary with metadata

4. **format_detection_result(...)**
   - Formats results into FileMetadata object
   - Combines file type, language, metadata

5. **detect_format_and_language(file_path)**
   - Main orchestrator function
   - Coordinates all detection steps
   - Returns: FileMetadata object

### Dependencies
- **polyglot** - Language detection (**open source**, initial model download required, then offline)
- **pycld2** - Compact Language Detector 2 (**open source**)
- **python-magic** - MIME type detection (optional, **open source**)

### Input
- File path (Path object)

### Output
- `FileMetadata` object containing:
  - file_path
  - file_type
  - language
  - file_size
  - encoding
  - created_at
  - modified_at

---

## ⚙️ Agent 2: Optimizer (`agents/optimizer.py`)

**Implementation type**: All optimization logic (extraction routing, cleaning, deduplication, chunking) is **hard coded by us**, built on top of **open source parsers**.

**Local/privacy fit**: Fully **local**, only reads/writes local files; **no external services** and **no data sharing**.

### Text Extraction Functions
1. **extract_text_pdf(file_path)**
   - Uses pdfplumber
   - Extracts text from all pages

2. **extract_text_docx(file_path)**
   - Uses python-docx
   - Extracts text from paragraphs

3. **extract_text_csv(file_path, encoding)**
   - Reads CSV files
   - Handles encoding

4. **extract_text_html(file_path, encoding)**
   - Uses BeautifulSoup
   - Extracts text from HTML/Notion exports

5. **extract_text(file_path, file_type, encoding)**
   - Routes to appropriate extractor
   - Main extraction orchestrator

### Processing Functions
6. **clean_text(text)**
   - Removes null bytes
   - Normalizes line endings
   - Removes extra whitespace
   - Fixes encoding issues

7. **detect_duplicates(texts, threshold)**
   - Uses thefuzz for fuzzy matching
   - Default threshold: 0.85
   - Returns list of duplicate indices

8. **chunk_text(text, chunk_size, chunk_overlap)**
   - Default chunk size: 500 characters
   - Default overlap: 50 characters
   - Splits by sentences
   - Returns list of text chunks

9. **optimize_data(file_path, metadata)**
   - Main orchestrator function
   - Coordinates extraction, cleaning, deduplication, chunking
   - Creates TextChunk objects
   - Returns: ProcessedDocument object

### Dependencies
- **pdfplumber** - PDF text extraction (**open source**, local)
- **python-docx** - DOCX parsing (**open source**, local)
- **beautifulsoup4** - HTML parsing (**open source**, local)
- **thefuzz** - Fuzzy duplicate detection (**open source**, local)

### Input
- File path (Path)
- FileMetadata object

### Output
- `ProcessedDocument` object containing:
  - file_path
  - metadata (FileMetadata)
  - raw_text
  - cleaned_text
  - chunks (List[TextChunk])
  - duplicates_removed (int)

---

## 🧬 Agent 3: Embedding Generator (`agents/embeddings.py`)

**Implementation type**: Chunk collection, DB wiring, and storage flow are **hard coded by us**; embeddings and vector storage use **open source** components.

**LLM/model usage**: Uses a **SentenceTransformers embedding model** (`intfloat/multilingual-e5-base`) – transformer encoder, not a chat LLM.

**Local/privacy fit**: After the embedding model is downloaded the first time, all inference and ChromaDB writes are **local and offline**, with telemetry disabled; **no data is sent outside**.

### Functions (4 functions)
1. **load_embedding_model()**
   - Loads sentence-transformers model
   - Model: `intfloat/multilingual-e5-base`
   - Caches model globally

2. **get_vector_db()**
   - Gets or creates ChromaDB instance
   - Creates collection if needed
   - Returns: database client and collection

3. **generate_embeddings_for_chunks(chunks)**
   - Generates embeddings for text chunks
   - Uses loaded model
   - Returns: List of embedding vectors

4. **store_embeddings(chunks, embeddings)**
   - Stores embeddings in ChromaDB
   - Includes metadata (document_id, language, file_type, indices)

5. **generate_embeddings(documents)**
   - Main orchestrator function
   - Collects all chunks from documents
   - Generates and stores embeddings
   - Returns: Boolean (success/failure)

### Dependencies
- **sentence-transformers** - Multilingual embeddings (**open source**, one-time model download, then offline)
- **chromadb** - Vector database (**open source**, `anonymized_telemetry=False` disables phone-home)

### Input
- List of ProcessedDocument objects

### Output
- Boolean: success/failure
- Vector database stored at `config.VECTOR_DB_PATH`

---

## 📊 Agent 4: Evaluator (`agents/evaluator.py`)

**Implementation type**: Evaluation strategy, metrics, and prompt construction are **hard coded by us**; similarity and LLM calls use **open source** components.

**LLM/model usage**:
- Uses **SentenceTransformers** (`paraphrase-multilingual-MiniLM-L12-v2`) for relevance scoring (encoder model).
- Uses **Ollama** to call a **local LLM** (default `"mistral"`) via `query_ollama`.

**Local/privacy fit**: After models are pulled (SentenceTransformers + Ollama), all evaluation is **local** and uses `http://localhost:11434` only; **no prompts or user data leave the machine**.

### Functions (7 functions)
1. **load_similarity_model()**
   - Loads model for similarity computation
   - Model: `paraphrase-multilingual-MiniLM-L12-v2`

2. **query_ollama(prompt, model)**
   - Queries Ollama LLM
   - Default model: "mistral"
   - Returns: Response with metrics (latency, tokens)

3. **count_tokens_approx(text)**
   - Approximates token count
   - Estimate: 1 token ≈ 4 characters

4. **compute_relevance_score(text1, text2)**
   - Computes cosine similarity
   - Uses sentence-transformers embeddings

5. **evaluate_raw_data(documents, query)**
   - Evaluates raw, unoptimized data
   - Combines all raw text
   - Queries LLM with full context

6. **evaluate_optimized_data(documents, query)**
   - Evaluates optimized data using RAG
   - Queries vector database for relevant chunks
   - Uses top 5 results as context

7. **evaluate_pipeline(documents, query, ground_truth)**
   - Main orchestrator function
   - Compares raw vs optimized performance
   - Calculates improvements:
     - Token reduction percentage
     - Latency reduction percentage
     - Relevance increase (if ground truth provided)

### Dependencies
- **ollama** - Local LLM integration (**open source**, local HTTP on `localhost`)
- **sentence-transformers** - For relevance calculations (**open source**, one-time model download, then offline)
- **numpy** - Numerical operations (**open source**, local)

### Input
- List of ProcessedDocument objects
- Query string
- Ground truth (optional)

### Output
- Dictionary with evaluation metrics:
  - raw: Raw data evaluation results
  - optimized: Optimized data evaluation results
  - improvement: Comparison metrics

---

## 📦 Data Models (`models.py`)

**Implementation type**: All data model structures (`FileMetadata`, `TextChunk`, `ProcessedDocument`, `PipelineResults`) are **hard coded by us**.

**Local/privacy fit**: Pure in-memory/file-structure definitions, **no external dependencies** and **no data sharing**.

### FileMetadata (dataclass)
- `file_path: Path` - Path to file
- `file_type: str` - Detected file type
- `language: str` - Detected language code
- `file_size: int` - File size in bytes
- `encoding: Optional[str]` - File encoding
- `created_at: Optional[str]` - Creation timestamp
- `modified_at: Optional[str]` - Modification timestamp

### TextChunk (dataclass)
- `text: str` - Chunk text content
- `chunk_id: str` - Unique chunk identifier
- `document_id: str` - Parent document identifier
- `language: str` - Language of chunk
- `start_index: int` - Start position in original text
- `end_index: int` - End position in original text
- `metadata: Dict[str, Any]` - Additional metadata

### ProcessedDocument (dataclass)
- `file_path: Path` - Path to original file
- `metadata: FileMetadata` - File metadata
- `raw_text: str` - Original extracted text
- `cleaned_text: str` - Cleaned text (after deduplication)
- `chunks: List[TextChunk]` - List of text chunks
- `duplicates_removed: int` - Number of duplicate chunks removed

### PipelineResults (dataclass)
- `documents: List[ProcessedDocument]` - Processed documents
- `total_files: int` - Total number of files
- `processed_files: int` - Successfully processed files
- `failed_files: int` - Failed files
- `total_chunks: int` - Total number of chunks
- `embeddings_generated: bool` - Whether embeddings were generated
- `vector_db_path: Optional[Path]` - Path to vector database
- `evaluation_metrics: Optional[Dict[str, Any]]` - Evaluation metrics

---

## 🔧 Configuration (`config.py`)

**Implementation type**: All configuration values (paths, model names, thresholds) are **hard coded by us**.

**Local/privacy fit**: Controls only local paths and localhost URLs; **no remote endpoints** other than `http://localhost:11434` for Ollama.

### Directories
- `BASE_DIR` - Project base directory
- `INPUT_DIR` - Input files directory (`input/`)
- `OUTPUT_DIR` - Output directory (`output/`)
- `TEST_DATA_DIR` - Test data directory (`tests/test_data/`)

### Embedding Settings
- `EMBEDDING_MODEL` - "intfloat/multilingual-e5-base"

### Chunking Settings
- `CHUNK_SIZE` - 500 characters
- `CHUNK_OVERLAP` - 50 characters

### Duplicate Detection
- `FUZZY_MATCH_THRESHOLD` - 0.85

### Vector Database
- `VECTOR_DB_PATH` - `output/chroma_db`
- `VECTOR_DB_COLLECTION` - "edalo_documents"

### Ollama Settings
- `OLLAMA_BASE_URL` - "http://localhost:11434"
- `OLLAMA_MODEL` - "mistral"

### Output Files
- `OUTPUT_JSONL` - `output/output.jsonl`
- `OUTPUT_SQLITE` - `output/output.db`
- `OUTPUT_METRICS` - `output/metrics.json`

### Logging
- `LOG_LEVEL` - Environment variable or "INFO"
- `LOG_FILE` - `output/edalo.log`

---

## 📤 Output Writers (`output.py`)

**Implementation type**: All output-writing logic (JSONL, SQLite, metrics) is **hard coded by us**, using standard/local libraries.

**Local/privacy fit**: Writes only to the local filesystem under `output/`; **no external sinks**.

### Functions (4 functions)
1. **write_jsonl(results, output_path)**
   - Writes chunks to JSONL format
   - One JSON object per line
   - Includes chunk metadata

2. **write_sqlite(results, output_path)**
   - Creates SQLite database
   - Two tables: `documents` and `chunks`
   - Foreign key relationships

3. **generate_metrics_report(results, output_path)**
   - Generates metrics JSON file
   - Includes:
     - File counts
     - Chunk statistics
     - Language distribution
     - Duplicate removal stats
     - Evaluation metrics (if available)

4. **write_outputs(results)**
   - Main orchestrator function
   - Writes all output formats
   - Returns dictionary of output paths

---

## 🖥️ CLI Interface (`cli.py`)

**Implementation type**: Argument parsing, validation, and CLI UX are **hard coded by us**.

**Local/privacy fit**: CLI only orchestrates local pipeline calls; **no network calls** from `cli.py` itself.

### Functions (3 functions)
1. **parse_arguments()**
   - Parses command-line arguments
   - Arguments:
     - `--input` (required): Input file or directory
     - `--no-embeddings`: Skip embedding generation
     - `--evaluate`: Run evaluation
     - `--query`: Query for evaluation
     - `--ground-truth`: Ground truth for evaluation

2. **validate_input_path(input_path)**
   - Validates input path exists
   - Exits with error if invalid

3. **main()**
   - Main CLI entry point
   - Coordinates argument parsing, validation, pipeline execution
   - Prints results summary

### Usage Examples
```bash
# Basic usage
python cli.py --input /path/to/files

# Skip embeddings
python cli.py --input /path/to/files --no-embeddings

# With evaluation
python cli.py --input /path/to/files --evaluate --query "What is the main topic?"
```

---

## 🔄 Pipeline Orchestrator (`pipeline.py`)

**Implementation type**: File collection, per-file processing, and overall pipeline orchestration are **hard coded by us**.

**Local/privacy fit**: Orchestrates only local agents and file operations; **no external APIs**.

### Functions (3 functions)
1. **collect_files(input_path)**
   - Collects all supported files
   - Handles both file and directory inputs
   - Recursive directory scanning

2. **process_single_file(file_path)**
   - Processes a single file through pipeline
   - Runs Detector → Optimizer
   - Returns: ProcessedDocument or None

3. **run_pipeline(input_path, generate_embeddings_flag, evaluate, query, ground_truth)**
   - Main pipeline orchestrator
   - Process:
     1. Collect files
     2. Process each file (Detector → Optimizer)
     3. Generate embeddings (optional)
     4. Run evaluation (optional)
   - Returns: PipelineResults

---

## 🛠️ Utilities (`utils.py`)

**Implementation type**: Logging setup is **hard coded by us**.

**Local/privacy fit**: Logs to local file and stdout only; **no remote logging endpoints**.

### Functions (1 function)
1. **setup_logging()**
   - Configures logging
   - File handler: `output/edalo.log`
   - Console handler: stdout
   - Configurable log level

---

## 📋 Dependencies (`requirements.txt`)

**Overall note**: All listed libraries are **open source**. Some (Polyglot models, SentenceTransformers models, Ollama models) require a **one-time download**, but once installed they run **fully local without sending user data outside**.

### Core Dependencies
- **polyglot** (16.7.4) - Language detection
- **pycld2** (0.42) - Compact Language Detector 2
- **pyicu** (2.16) - ICU library bindings
- **morfessor** (2.0.6) - Morphological segmentation

### File Processing
- **pdfplumber** (0.11.0) - PDF text extraction
- **python-docx** (1.1.0) - DOCX parsing
- **beautifulsoup4** (4.12.3) - HTML parsing
- **lxml** (5.1.0) - XML/HTML parser
- **python-magic** (0.4.27) - MIME type detection

### ML/AI
- **sentence-transformers** (2.3.1) - Multilingual embeddings
- **chromadb** (0.4.22) - Vector database
- **ollama** (0.1.7) - Local LLM integration
- **numpy** - Numerical operations

### Text Processing
- **thefuzz** (0.22.1) - Fuzzy string matching
- **python-Levenshtein** (0.23.0) - String similarity

### Visualization/Reporting
- **matplotlib** (3.8.2) - Plotting
- **reportlab** (4.0.7) - PDF generation

### System Dependencies
- **libicu-dev** (Linux) / **icu4c** (macOS) - For Polyglot
- **libmagic** - For python-magic
- **Ollama** - Local LLM server

---

## 🧪 Testing

### Test Data Generator (`tests/generate_test_data.py`)
- Generates multilingual test files
- Formats: PDF, DOCX, CSV, TXT
- Languages: FR, DE, EN, LU

### Test Pipeline (`tests/test_pipeline.py`)
- Tests full pipeline execution
- Verifies output formats
- Checks metrics generation

### Test Files
- Located in `tests/test_data/`
- Multiple languages and formats
- Used for validation

---

## 📊 Output Formats

### JSONL (`output.jsonl`)
- One JSON object per line
- Contains chunk data with metadata
- Easy to stream and process

### SQLite (`output.db`)
- Relational database
- Tables: `documents`, `chunks`
- Foreign key relationships
- Queryable with SQL

### ChromaDB (`output/chroma_db/`)
- Vector database
- Stores embeddings for semantic search
- Enables RAG (Retrieval-Augmented Generation)

### Metrics (`metrics.json`)
- Performance metrics
- Evaluation results (if evaluated)
- Language distribution
- Duplicate removal statistics

### Logs (`output/edalo.log`)
- Application logs
- Error tracking
- Processing information

---

## 🎯 Code Constraints

### Function Constraints
- Maximum 25 lines per function
- Maximum 5 functions per file
- Use while loops instead of for loops
- No ternaries (use if/else statements)

### Architecture Constraints
- 4-agent system (Detector, Optimizer, Embeddings, Evaluator)
- Sequential processing through pipeline
- Local processing only (no cloud APIs)
- Multilingual support required

### Code Quality
- All functions must have docstrings
- Document parameters and return values
- Use descriptive variable names
- Keep functions focused on single responsibility

---

## 🌍 Language Support

### Detected Languages
- French (fr)
- German (de)
- English (en)
- Luxembourgish (lb)
- 130+ languages supported by Polyglot

### Language Models
- Requires Polyglot language models
- Download commands:
  ```bash
  polyglot download LANG:fr
  polyglot download LANG:de
  polyglot download LANG:en
  polyglot download LANG:lb
  ```

---

## 🔐 Error Handling

### Graceful Degradation
- Missing dependencies: Fall back to alternatives
- File processing errors: Log error, continue
- Language detection failures: Use "unknown"
- Encoding issues: Try multiple encodings

### Logging
- All errors logged to `output/edalo.log`
- Configurable log level via environment variable
- Includes timestamps and detailed error messages

---

## 📈 Data Flow

```
Input Files
    ↓
[collect_files] → List[Path]
    ↓
[process_single_file] (for each file)
    ↓
[detect_format_and_language] → FileMetadata
    ↓
[optimize_data] → ProcessedDocument
    ↓
[generate_embeddings] (optional) → Vector DB
    ↓
[evaluate_pipeline] (optional) → Metrics
    ↓
[write_outputs] → JSONL + SQLite + Metrics
```

---

## 🔄 Execution Flow

1. **CLI Entry Point** (`cli.py:main()`)
   - Parse arguments
   - Validate input
   - Call pipeline

2. **Pipeline Orchestration** (`pipeline.py:run_pipeline()`)
   - Collect files
   - Process files sequentially
   - Generate embeddings (if requested)
   - Run evaluation (if requested)

3. **Agent Execution** (Sequential)
   - Agent 1: Detection
   - Agent 2: Optimization
   - Agent 3: Embeddings (optional)
   - Agent 4: Evaluation (optional)

4. **Output Generation** (`output.py:write_outputs()`)
   - Write JSONL
   - Write SQLite
   - Write metrics

5. **Summary Display** (`cli.py:main()`)
   - Print statistics
   - Show output paths
   - Display evaluation metrics (if available)

---

## 🎨 Features

### Core Features
- ✅ Multilingual support (130+ languages)
- ✅ Multiple file formats (PDF, DOCX, CSV, TXT, HTML, JSON)
- ✅ Data optimization (cleaning, deduplication, chunking)
- ✅ Embedding generation for semantic search
- ✅ Local processing (no external APIs)
- ✅ LLM integration via Ollama

### Advanced Features
- ✅ Vector database for RAG
- ✅ Performance evaluation
- ✅ Multiple output formats
- ✅ Comprehensive logging
- ✅ Error handling and recovery

---

## 📚 Documentation Files

- **README.md** - Project overview and usage
- **ARCHITECTURE.md** - Detailed architecture documentation
- **WORKFLOW.md** - Development workflow and change management
- **CHANGELOG.md** - Project changelog and version history
- **MINDMAP.md** - This comprehensive mindmap

---

## 🔗 Key Relationships

### Module Dependencies
```
cli.py → pipeline.py → agents/*
pipeline.py → models.py
agents/* → models.py, config.py
output.py → models.py, config.py
utils.py → config.py
```

### Data Flow
```
FileMetadata → ProcessedDocument → PipelineResults
TextChunk (part of ProcessedDocument)
```

### Agent Dependencies
```
Detector → FileMetadata
Optimizer → ProcessedDocument (uses FileMetadata)
Embeddings → Vector DB (uses ProcessedDocument)
Evaluator → Metrics (uses ProcessedDocument, Vector DB)
```

---

## 🎓 Use Cases

1. **Educational Content Processing**
   - Process multilingual educational materials
   - Extract and structure content
   - Prepare for LLM consumption

2. **Semantic Search**
   - Generate embeddings for documents
   - Enable similarity search
   - RAG for question answering

3. **Data Cleaning**
   - Remove duplicates
   - Clean and normalize text
   - Structure unstructured data

4. **Performance Evaluation**
   - Compare raw vs optimized data
   - Measure token reduction
   - Evaluate latency improvements

---

## 🚀 Future Enhancements

### Potential Improvements
- Support for more file formats
- Improved duplicate detection algorithms
- Better chunking strategies
- Additional evaluation metrics
- Performance optimizations
- Batch processing improvements

### Experimentation & Configurability (LLM-Driven Optimization Loop)

- **Current state (done)**
  - Detector (Polyglot-based) and Optimizer are **deterministic** for a given input file:
    - Always produce the same `FileMetadata` and `ProcessedDocument` for the same raw file.
    - Parameters (chunk size, overlap, fuzziness) are **hard coded in `config.py`**.
  - Evaluator already:
    - Takes pipeline output and computes **metrics** (latency, token usage, relevance).
    - Can be re-run with **different LLM models** (via `config.OLLAMA_MODEL`) to compare behavior.
- **Intended value**
  - Use metrics from Evaluator to:
    - Compare **different models** and **different preprocessing settings**.
    - Identify **weak spots** in outputs (e.g., low relevance, too many tokens).
    - Iterate towards an **improved “version”** of the pipeline configuration.
  - This requires **freedom in inputs**:
    - Ability to vary LLM model, prompts, chunking and deduplication parameters **without code changes**.

### Next Steps (design-level)

- **Expose configurable knobs** (still to do)
  - Make key parameters configurable at runtime:
    - Detector: language-detection options (e.g., minimum text length, fallback behavior).
    - Optimizer: `CHUNK_SIZE`, `CHUNK_OVERLAP`, duplicate threshold (`FUZZY_MATCH_THRESHOLD`), cleaning options.
    - Evaluator: number of retrieved chunks, alternative LLM models, prompt templates.
  - Add a **configuration layer** (e.g., CLI flags or a config file) to adjust these without modifying code.
- **Tighter optimization loop** (still to do)
  - Automate experiments:
    - Run pipeline multiple times with **different parameter sets** and **different LLM models**.
    - Compare resulting metrics and highlight **best-performing configurations**.
  - Feed insights back into:
    - Updated recommendations for `config.py` defaults.
    - Suggested “profiles” (e.g., *max quality*, *max speed*, *max privacy*).

### Maintenance
- Regular dependency updates
- Code refactoring for maintainability
- Documentation updates
- Test coverage improvements

---

## 📝 Summary

**Edalo** is a local Python pipeline that processes multilingual educational files through a 4-agent system:

1. **Detector** - Identifies file format and language
2. **Optimizer** - Extracts, cleans, and chunks text
3. **Embeddings** - Generates multilingual embeddings for semantic search
4. **Evaluator** - Measures performance and relevance

The pipeline transforms unstructured data into clean, structured output optimized for LLM consumption, with support for multiple file formats, 130+ languages, and local processing without external API dependencies.

