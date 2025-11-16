"""Test framework for Edalo pipeline."""

import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
import pipeline
import output

logger = logging.getLogger(__name__)


def get_ground_truth():
    """Get ground truth for evaluation."""
    return "Students express concerns about too much theoretical content and insufficient practical exercises. They want more hands-on activities and fewer lectures."


def get_test_query():
    """Get test query for evaluation."""
    return "What are the main issues students mention in their feedback?"


def run_raw_data_test(test_data_dir: Path):
    """Run test with raw data."""
    logger.info("Running raw data test")
    results = pipeline.run_pipeline(
        input_path=test_data_dir, # path to the test data directory
        generate_embeddings_flag=False, # flag to generate embeddings
        evaluate=False, # flag to evaluate the pipeline
    )
    return results # returns the results of the pipeline


def run_optimized_data_test(test_data_dir: Path, query: str, ground_truth: str):
    """Run test with optimized data."""
    logger.info("Running optimized data test")
    results = pipeline.run_pipeline(
        input_path=test_data_dir,
        generate_embeddings_flag=True,
        evaluate=True,
        query=query,
        ground_truth=ground_truth,
    )
    return results


def generate_visualization(metrics: dict, output_path: Path):
    """Generate visualization of metrics."""
    logger.info(f"Generating visualization: {output_path}")
    if "improvement" not in metrics:
        logger.warning("No improvement metrics to visualize")
        return
    improvement = metrics["improvement"]
    categories = []
    values = []
    if "token_reduction_percent" in improvement:
        categories.append("Token Reduction")
        values.append(improvement["token_reduction_percent"])
    if "latency_reduction_percent" in improvement:
        categories.append("Latency Reduction")
        values.append(improvement["latency_reduction_percent"])
    if "relevance_increase" in improvement:
        categories.append("Relevance Increase")
        values.append(improvement["relevance_increase"] * 100)
    if not categories:
        logger.warning("No metrics to visualize")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_ylabel("Percentage")
    ax.set_title("Pipeline Performance Improvements")
    ax.set_ylim([0, max(values) * 1.2 if values else 100])
    bar_index = 0
    while bar_index < len(bars):
        bar = bars[bar_index]
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
        )
        bar_index += 1
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Visualization saved to {output_path}")


def run_tests():
    """Run all tests."""
    logger.info("Starting test suite")
    test_data_dir = config.TEST_DATA_DIR
    if not test_data_dir.exists():
        logger.error(f"Test data directory does not exist: {test_data_dir}")
        logger.info("Please run tests/generate_test_data.py first")
        return
    query = get_test_query()
    ground_truth = get_ground_truth()
    logger.info("Running optimized data test with evaluation")
    results = run_optimized_data_test(test_data_dir, query, ground_truth)
    outputs = output.write_outputs(results)
    if results.evaluation_metrics:
        metrics = results.evaluation_metrics
        viz_path = config.OUTPUT_DIR / "metrics_visualization.png"
        generate_visualization(metrics, viz_path)
        print("\n" + "=" * 50)
        print("Test Results")
        print("=" * 50)
        print(f"Processed files: {results.processed_files}")
        print(f"Total chunks: {results.total_chunks}")
        print("\nEvaluation Metrics:")
        if "raw" in metrics:
            raw = metrics["raw"]
            print(f"  Raw data - Tokens: {raw.get('total_tokens', 0)}, Latency: {raw.get('latency', 0):.2f}s")
        if "optimized" in metrics:
            optimized = metrics["optimized"]
            print(f"  Optimized data - Tokens: {optimized.get('total_tokens', 0)}, Latency: {optimized.get('latency', 0):.2f}s")
        if "improvement" in metrics:
            improvement = metrics["improvement"]
            print("\nImprovements:")
            if "token_reduction_percent" in improvement:
                print(f"  Token reduction: {improvement['token_reduction_percent']:.2f}%")
            if "latency_reduction_percent" in improvement:
                print(f"  Latency reduction: {improvement['latency_reduction_percent']:.2f}%")
            if "relevance_increase" in improvement:
                print(f"  Relevance increase: {improvement['relevance_increase']:.4f}")
        print(f"\nVisualization: {viz_path}")
        print("=" * 50)
    else:
        logger.warning("No evaluation metrics available")
    logger.info("Test suite completed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_tests()

