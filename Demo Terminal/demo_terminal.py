"""Interactive terminal demo for the Edalo 4-agent pipeline.

This demo uses existing pipeline outputs to walk through:
1) Detector
2) Optimizer
3) Embeddings
4) Evaluator
"""

import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.detector import detect_format_and_language


def load_metrics() -> dict:
    """Load metrics from the metrics.json file.

    Returns:
        dict: Parsed metrics dictionary, or an empty dict on failure.
    """
    metrics_path = Path("output") / "metrics.json"
    try:
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except Exception:
        return {}


def load_sample_chunks(max_chunks: int = 6) -> list:
    """Load a small set of sample chunks from output.jsonl.

    Args:
        max_chunks (int): Maximum number of chunks to load.

    Returns:
        list: List of chunk dicts for the demo.
    """
    chunks = []
    jsonl_path = Path("output") / "output.jsonl"
    try:
        with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
            all_lines = jsonl_file.readlines()
        total_lines = len(all_lines)
        if total_lines == 0:
            return []
        indices = list(range(total_lines))
        random.shuffle(indices)
        index = 0
        while index < total_lines:
            if len(chunks) >= max_chunks:
                break
            line = all_lines[indices[index]]
            chunk = json.loads(line)
            chunks.append(chunk)
            index += 1
    except Exception:
        return []
    return chunks


def print_header(title: str) -> None:
    """Print a formatted section header.

    Args:
        title (str): Title to display.
    """
    separator = "=" * 60
    print("\n" + separator)
    print(title)
    print(separator)


def run_steps(metrics: dict, chunks: list) -> None:
    """Run the interactive step-by-step terminal demo.

    Args:
        metrics (dict): Metrics loaded from metrics.json.
        chunks (list): Sample chunks loaded from output.jsonl.
    """
    print_header("Edalo 4-Agent Demo (Terminal)")
    total_files = metrics.get("total_files", 0)
    total_chunks = metrics.get("total_chunks", 0)
    print(f"Files: {total_files} | Chunks: {total_chunks}")
    languages = metrics.get("languages", {})
    if languages:
        print("Languages:", languages)
    if chunks:
        seen_paths = []
        file_paths = []
        index = 0
        while index < len(chunks):
            if len(file_paths) >= 6:
                break
            path_value = chunks[index].get("file_path", "")
            if path_value and path_value not in seen_paths:
                seen_paths.append(path_value)
                file_paths.append(path_value)
            index += 1
        if file_paths:
            print("\nExample processed files:")
            path_index = 0
            while path_index < len(file_paths):
                print(f"- {file_paths[path_index]}")
                path_index += 1
    input("\nPress Enter to start with Agent 1 - Detector...")
    print_header("Agent 1 - Detector (Format & Language)")
    sample_files = [
        Path("tests") / "test_data" / "feedback_fr_comprehensive.pdf",
        Path("tests") / "test_data" / "feedback_en_comprehensive.pdf",
        Path("tests") / "test_data" / "feedback_de_comprehensive.pdf",
        Path("tests") / "test_data" / "feedback_lb_comprehensive.pdf",
    ]
    sample_file = random.choice(sample_files)
    try:
        detected = detect_format_and_language(sample_file)
        print("Sample file:", detected.file_path)
        print("Detected type:", detected.file_type)
        print("Detected language:", detected.language)
    except Exception:
        if chunks:
            first_chunk = chunks[0]
            file_path = first_chunk.get("file_path", "")
            language = first_chunk.get("language", "unknown")
            metadata = first_chunk.get("metadata", {})
            file_type = metadata.get("file_type", "unknown")
            print("Sample file:", file_path)
            print("Detected type:", file_type)
            print("Detected language:", language)
    input("\nPress Enter to continue to Agent 2 - Optimizer...")
    print_header("Agent 2 - Optimizer (Clean, Deduplicate, Chunk)")
    duplicates_removed = metrics.get("total_duplicates_removed", 0)
    print("Duplicates removed across documents:", duplicates_removed)
    if chunks:
        print("\nSample optimized chunk text:")
        sample_index = random.randint(0, len(chunks) - 1)
        sample_text = chunks[sample_index].get("text", "")
        preview = sample_text[:400]
        print(preview)
    input("\nPress Enter to continue to Agent 3 - Embeddings...")
    print_header("Agent 3 - Embeddings (Vectorization)")
    vector_db_path = metrics.get("vector_db_path", "")
    print("Vector database path:", vector_db_path)
    if chunks:
        print("Example chunks stored as vectors:", len(chunks))
        random.shuffle(chunks)
        chunk_index = 0
        while chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            print(f"- {chunk.get('chunk_id', '')} [{chunk.get('language', '')}]")
            chunk_index += 1
    input("\nPress Enter to continue to Agent 4 - Evaluator...")
    print_header("Agent 4 - Evaluator (Prove Impact)")
    evaluation = metrics.get("evaluation", {})
    improvement = evaluation.get("improvement", {})
    token_reduction = improvement.get("token_reduction_percent", 0.0)
    latency_reduction = improvement.get("latency_reduction_percent", 0.0)
    relevance_increase = improvement.get("relevance_increase", 0.0)
    print("Token reduction (%):", f"{token_reduction:.2f}")
    print("Latency reduction (%):", f"{latency_reduction:.2f}")
    print("Relevance increase:", f"{relevance_increase:.3f}")
    print("\nDemo complete. Edalo transforms messy feedback into AI-ready data.")


def main() -> None:
    """Entry point for the terminal demo."""
    metrics = load_metrics()
    chunks = load_sample_chunks()
    run_steps(metrics, chunks)


if __name__ == "__main__":
    main()


