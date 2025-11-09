"""Agents package for Edalo pipeline."""

from agents.detector import detect_format_and_language
from agents.optimizer import optimize_data
from agents.embeddings import generate_embeddings
from agents.evaluator import evaluate_pipeline

__all__ = [
    "detect_format_and_language",
    "optimize_data",
    "generate_embeddings",
    "evaluate_pipeline",
]

