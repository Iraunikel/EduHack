# Changelog

All notable changes to the Edalo project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- WORKFLOW.md: Sustainable workflow documentation
- CHANGELOG.md: Change tracking document
- ARCHITECTURE.md: Detailed architecture documentation
- .cursorrules: Project-specific coding rules
- Luxembourgish (lb) language detection support via Polyglot

### Changed
- Reset project to first commit state (4fbb45e) to establish clean baseline
- Established sustainable workflow for future changes
- Updated README.md with links to new documentation
- Replaced langdetect with Polyglot for language detection
  - Polyglot supports 130+ languages including Luxembourgish (lb)
  - Fully offline/local processing - no cloud services or data sharing
  - Requires language model download: `polyglot download LANG:lb`
  - May require system ICU library installation (e.g., `brew install icu4c` on macOS)

### Removed
- cursor_optimizing_multilingual_educatio.md: Removed conversation history file
- tests/EVALUATION_METHODOLOGY.md: Removed (was added after first commit)
- tests/TEST_REPORT.md: Removed (was added after first commit)
- langdetect dependency (replaced with Polyglot)

### Notes
- First commit contains some constraint violations:
  - `agents/optimizer.py`: 9 functions (exceeds 5-function limit)
  - `agents/evaluator.py`: 7 functions (exceeds 5-function limit)
- These violations are documented and will be addressed in future refactoring
- Workflow established to prevent future constraint violations

## [1.0.0] - 2025-11-09

### Added
- Initial commit with full pipeline implementation
- Agent 1: Format & Language Detector
- Agent 2: Optimizer (text extraction, cleaning, chunking)
- Agent 3: Embedding Generator (multilingual embeddings)
- Agent 4: Evaluator (Ollama integration, metrics)
- Pipeline orchestrator
- CLI interface
- Data models (FileMetadata, TextChunk, ProcessedDocument, PipelineResults)
- Output writers (JSONL, SQLite, ChromaDB)
- Test data generator
- Test pipeline runner
- Configuration system
- Logging infrastructure
- README.md with installation and usage instructions
- Planning documents

### Features
- Multilingual support (FR, DE, EN, LU, etc.)
- Multiple file format support (PDF, DOCX, CSV, TXT, HTML, JSON)
- Data optimization (duplicate removal, cleaning, chunking)
- Embedding generation for semantic search
- Local processing (no external APIs)
- LLM integration via Ollama
- Evaluation metrics (tokens, latency, relevance)

### Constraints
- Functions max 25 lines
- Files max 5 functions
- Use while loops instead of for loops
- No ternaries
- Local execution only

