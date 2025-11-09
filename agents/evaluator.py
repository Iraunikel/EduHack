"""Agent 4: Evaluator - Measures performance and relevance."""

import logging
import time
from typing import Dict, List, Optional, Any

import ollama
from sentence_transformers import SentenceTransformer
import numpy as np

import config
from models import ProcessedDocument

logger = logging.getLogger(__name__)

_model = None


def load_similarity_model() -> SentenceTransformer:
    """Load model for similarity computation."""
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def query_ollama(prompt: str, model: str = None) -> Dict[str, Any]:
    """Query Ollama LLM and return response with metrics."""
    if model is None:
        model = config.OLLAMA_MODEL
    import os
    base_url = config.OLLAMA_BASE_URL
    os.environ.setdefault("OLLAMA_HOST", base_url.replace("http://", "").replace("https://", ""))
    start_time = time.time()
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
        )
        end_time = time.time()
        latency = end_time - start_time
        response_text = response.get("response", "")
        token_count = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
        return {
            "response": response_text,
            "latency": latency,
            "tokens": token_count,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return {
            "response": "",
            "latency": 0,
            "tokens": 0,
            "success": False,
            "error": str(e),
        }


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: 1 token ≈ 4 characters)."""
    return len(text) // 4


def compute_relevance_score(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    model = load_similarity_model()
    embeddings = model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def evaluate_raw_data(documents: List[ProcessedDocument], query: str) -> Dict[str, Any]:
    """Evaluate raw data by querying LLM directly."""
    logger.info("Evaluating raw data")
    combined_text = ""
    doc_index = 0
    while doc_index < len(documents):
        doc = documents[doc_index]
        combined_text += doc.raw_text + "\n\n"
        doc_index += 1
    prompt = f"{query}\n\nContext:\n{combined_text}"
    result = query_ollama(prompt)
    token_count = count_tokens_approx(prompt)
    result["input_tokens"] = token_count
    result["total_tokens"] = token_count + result.get("tokens", 0)
    return result


def evaluate_optimized_data(documents: List[ProcessedDocument], query: str) -> Dict[str, Any]:
    """Evaluate optimized data using RAG."""
    logger.info("Evaluating optimized data")
    from agents.embeddings import get_vector_db, load_embedding_model
    
    _, collection = get_vector_db()
    model = load_embedding_model()
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )
    context_texts = []
    if results.get("documents") and len(results["documents"]) > 0:
        doc_index = 0
        while doc_index < len(results["documents"][0]):
            context_texts.append(results["documents"][0][doc_index])
            doc_index += 1
    context = "\n\n".join(context_texts)
    prompt = f"{query}\n\nContext:\n{context}"
    result = query_ollama(prompt)
    token_count = count_tokens_approx(prompt)
    result["input_tokens"] = token_count
    result["total_tokens"] = token_count + result.get("tokens", 0)
    result["context_chunks"] = len(context_texts)
    return result


def evaluate_pipeline(
    documents: List[ProcessedDocument],
    query: str,
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """Main evaluator agent: compares raw vs optimized data performance."""
    logger.info("Running pipeline evaluation")
    raw_results = evaluate_raw_data(documents, query)
    optimized_results = evaluate_optimized_data(documents, query)
    metrics = {
        "raw": raw_results,
        "optimized": optimized_results,
        "improvement": {},
    }
    if raw_results.get("success") and optimized_results.get("success"):
        token_reduction = (
            (raw_results["total_tokens"] - optimized_results["total_tokens"])
            / raw_results["total_tokens"]
            * 100
            if raw_results["total_tokens"] > 0
            else 0
        )
        latency_reduction = (
            (raw_results["latency"] - optimized_results["latency"])
            / raw_results["latency"]
            * 100
            if raw_results["latency"] > 0
            else 0
        )
        metrics["improvement"] = {
            "token_reduction_percent": token_reduction,
            "latency_reduction_percent": latency_reduction,
        }
        if ground_truth:
            raw_relevance = compute_relevance_score(raw_results["response"], ground_truth)
            optimized_relevance = compute_relevance_score(optimized_results["response"], ground_truth)
            metrics["raw"]["relevance"] = raw_relevance
            metrics["optimized"]["relevance"] = optimized_relevance
            metrics["improvement"]["relevance_increase"] = optimized_relevance - raw_relevance
    logger.info("Evaluation completed")
    return metrics

