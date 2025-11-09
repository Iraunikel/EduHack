"""Main pipeline orchestrator for Edalo."""

import logging
from pathlib import Path
from typing import List, Optional

import utils
from models import PipelineResults, ProcessedDocument
from agents.detector import detect_format_and_language
from agents.optimizer import optimize_data
from agents.embeddings import generate_embeddings
from agents.evaluator import evaluate_pipeline

logger = utils.setup_logging()


def collect_files(input_path: Path) -> List[Path]:
    """Collect all supported files from input path."""
    supported_extensions = {".pdf", ".docx", ".csv", ".txt", ".html", ".json"}
    files = []
    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            files.append(input_path)
    elif input_path.is_dir():
        file_list = list(input_path.rglob("*"))
        file_index = 0
        while file_index < len(file_list):
            file_path = file_list[file_index]
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
            file_index += 1
    return files


def process_single_file(file_path: Path) -> Optional[ProcessedDocument]:
    """Process a single file through the pipeline."""
    try:
        metadata = detect_format_and_language(file_path)
        document = optimize_data(file_path, metadata)
        return document
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None


def run_pipeline(
    input_path: Path,
    generate_embeddings_flag: bool = True,
    evaluate: bool = False,
    query: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> PipelineResults:
    """Main pipeline orchestrator."""
    logger.info(f"Starting pipeline for {input_path}")
    files = collect_files(input_path)
    logger.info(f"Found {len(files)} files to process")
    results = PipelineResults(total_files=len(files))
    documents = []
    file_index = 0
    while file_index < len(files):
        file_path = files[file_index]
        logger.info(f"Processing file {file_index + 1}/{len(files)}: {file_path}")
        document = process_single_file(file_path)
        if document:
            documents.append(document)
            results.processed_files += 1
            results.total_chunks += len(document.chunks)
        else:
            results.failed_files += 1
        file_index += 1
    results.documents = documents
    if generate_embeddings_flag and documents:
        logger.info("Generating embeddings")
        embeddings_success = generate_embeddings(documents)
        results.embeddings_generated = embeddings_success
        if embeddings_success:
            import config
            results.vector_db_path = config.VECTOR_DB_PATH
    if evaluate and query:
        logger.info("Running evaluation")
        evaluation_metrics = evaluate_pipeline(documents, query, ground_truth)
        results.evaluation_metrics = evaluation_metrics
    logger.info("Pipeline completed successfully")
    return results

