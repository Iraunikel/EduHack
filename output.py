"""Output writers for Edalo pipeline."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

import config
from models import PipelineResults, ProcessedDocument, TextChunk

logger = logging.getLogger(__name__)


def write_jsonl(results: PipelineResults, output_path: Path = None) -> Path:
    """Write pipeline results to JSONL format."""
    if output_path is None:
        output_path = config.OUTPUT_JSONL
    logger.info(f"Writing JSONL output to {output_path}")
    doc_index = 0
    with open(output_path, "w", encoding="utf-8") as f:
        while doc_index < len(results.documents):
            doc = results.documents[doc_index]
            chunk_index = 0
            while chunk_index < len(doc.chunks):
                chunk = doc.chunks[chunk_index]
                record = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "file_path": str(doc.file_path),
                    "text": chunk.text,
                    "language": chunk.language,
                    "metadata": chunk.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_index += 1
            doc_index += 1
    logger.info(f"Wrote {results.total_chunks} chunks to JSONL")
    return output_path


def write_sqlite(results: PipelineResults, output_path: Path = None) -> Path:
    """Write pipeline results to SQLite database."""
    if output_path is None:
        output_path = config.OUTPUT_SQLITE
    logger.info(f"Writing SQLite output to {output_path}")
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            file_path TEXT,
            file_type TEXT,
            language TEXT,
            file_size INTEGER,
            raw_text TEXT,
            cleaned_text TEXT,
            duplicates_removed INTEGER
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT,
            text TEXT,
            language TEXT,
            start_index INTEGER,
            end_index INTEGER,
            metadata TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )"""
    )
    conn.commit()
    doc_index = 0
    while doc_index < len(results.documents):
        doc = results.documents[doc_index]
        document_id = doc.chunks[0].document_id if doc.chunks else ""
        cursor.execute(
            """INSERT OR REPLACE INTO documents
                (document_id, file_path, file_type, language, file_size,
                 raw_text, cleaned_text, duplicates_removed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                str(doc.file_path),
                doc.metadata.file_type,
                doc.metadata.language,
                doc.metadata.file_size,
                doc.raw_text,
                doc.cleaned_text,
                doc.duplicates_removed,
            ),
        )
        chunk_index = 0
        while chunk_index < len(doc.chunks):
            chunk = doc.chunks[chunk_index]
            cursor.execute(
                """INSERT OR REPLACE INTO chunks
                    (chunk_id, document_id, text, language, start_index,
                     end_index, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.text,
                    chunk.language,
                    chunk.start_index,
                    chunk.end_index,
                    json.dumps(chunk.metadata),
                ),
            )
            chunk_index += 1
        doc_index += 1
    conn.commit()
    conn.close()
    logger.info(f"Wrote {results.processed_files} documents to SQLite")
    return output_path


def generate_metrics_report(results: PipelineResults, output_path: Path = None) -> Path:
    """Generate metrics report in JSON format."""
    if output_path is None:
        output_path = config.OUTPUT_METRICS
    logger.info(f"Generating metrics report to {output_path}")
    report = {
        "total_files": results.total_files,
        "processed_files": results.processed_files,
        "failed_files": results.failed_files,
        "total_chunks": results.total_chunks,
        "embeddings_generated": results.embeddings_generated,
        "vector_db_path": str(results.vector_db_path) if results.vector_db_path else None,
    }
    if results.evaluation_metrics:
        report["evaluation"] = results.evaluation_metrics
    languages = {}
    doc_index = 0
    while doc_index < len(results.documents):
        doc = results.documents[doc_index]
        lang = doc.metadata.language
        if lang in languages:
            languages[lang] += 1
        else:
            languages[lang] = 1
        doc_index += 1
    report["languages"] = languages
    total_duplicates = 0
    doc_index = 0
    while doc_index < len(results.documents):
        total_duplicates += results.documents[doc_index].duplicates_removed
        doc_index += 1
    report["total_duplicates_removed"] = total_duplicates
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Metrics report generated")
    return output_path


def write_outputs(results: PipelineResults) -> Dict[str, Path]:
    """Write all output formats."""
    logger.info("Writing all output formats")
    outputs = {
        "jsonl": write_jsonl(results),
        "sqlite": write_sqlite(results),
        "metrics": generate_metrics_report(results),
    }
    logger.info("All outputs written successfully")
    return outputs

