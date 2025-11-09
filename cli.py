"""CLI interface for Edalo pipeline."""

import argparse
import logging
import sys
from pathlib import Path

import utils
import pipeline
import output

logger = utils.setup_logging()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Edalo - Educational Data LLM Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory path",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after processing",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query for evaluation",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Ground truth text for evaluation",
    )
    return parser.parse_args()


def validate_input_path(input_path: str) -> Path:
    """Validate and return input path."""
    path = Path(input_path)
    if not path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    return path


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    input_path = validate_input_path(args.input)
    logger.info(f"Starting Edalo pipeline with input: {input_path}")
    generate_embeddings = not args.no_embeddings
    evaluate = args.evaluate
    query = args.query
    ground_truth = args.ground_truth
    if evaluate and not query:
        logger.error("--query is required when --evaluate is set")
        sys.exit(1)
    try:
        results = pipeline.run_pipeline(
            input_path=input_path,
            generate_embeddings_flag=generate_embeddings,
            evaluate=evaluate,
            query=query,
            ground_truth=ground_truth,
        )
        outputs = output.write_outputs(results)
        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print("=" * 50)
        print(f"Processed files: {results.processed_files}/{results.total_files}")
        print(f"Total chunks: {results.total_chunks}")
        print(f"Embeddings generated: {results.embeddings_generated}")
        print("\nOutput files:")
        for output_type, output_path in outputs.items():
            print(f"  {output_type}: {output_path}")
        if results.evaluation_metrics:
            print("\nEvaluation metrics:")
            metrics = results.evaluation_metrics
            if "improvement" in metrics:
                improvement = metrics["improvement"]
                if "token_reduction_percent" in improvement:
                    print(f"  Token reduction: {improvement['token_reduction_percent']:.2f}%")
                if "latency_reduction_percent" in improvement:
                    print(f"  Latency reduction: {improvement['latency_reduction_percent']:.2f}%")
                if "relevance_increase" in improvement:
                    print(f"  Relevance increase: {improvement['relevance_increase']:.4f}")
        print("=" * 50)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

