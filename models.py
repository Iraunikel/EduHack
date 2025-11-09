"""Data models for Edalo pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class FileMetadata:
    """Metadata for a processed file."""

    file_path: Path
    file_type: str
    language: str
    file_size: int
    encoding: Optional[str] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None


@dataclass
class TextChunk:
    """A chunk of processed text."""

    text: str
    chunk_id: str
    document_id: str
    language: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """A processed document with extracted text and metadata."""

    file_path: Path
    metadata: FileMetadata
    raw_text: str
    cleaned_text: str
    chunks: List[TextChunk] = field(default_factory=list)
    duplicates_removed: int = 0


@dataclass
class PipelineResults:
    """Results from the pipeline execution."""

    documents: List[ProcessedDocument] = field(default_factory=list)
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    embeddings_generated: bool = False
    vector_db_path: Optional[Path] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None

