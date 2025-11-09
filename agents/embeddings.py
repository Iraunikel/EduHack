"""Agent 3: Embedding Generator - Creates multilingual embeddings."""

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config
from models import ProcessedDocument, TextChunk

logger = logging.getLogger(__name__)

_model = None
_vector_db = None
_collection = None


def load_embedding_model() -> SentenceTransformer:
    """Load the multilingual embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def get_vector_db():
    """Get or create ChromaDB vector database."""
    global _vector_db, _collection
    if _vector_db is None:
        logger.info(f"Initializing vector database at {config.VECTOR_DB_PATH}")
        _vector_db = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            _collection = _vector_db.get_collection(config.VECTOR_DB_COLLECTION)
            logger.info("Using existing collection")
        except Exception:
            _collection = _vector_db.create_collection(
                name=config.VECTOR_DB_COLLECTION,
                metadata={"description": "Edalo document embeddings"},
            )
            logger.info("Created new collection")
    return _vector_db, _collection


def generate_embeddings_for_chunks(chunks: List[TextChunk]) -> List[List[float]]:
    """Generate embeddings for a list of text chunks."""
    model = load_embedding_model()
    texts = [chunk.text for chunk in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks")
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def store_embeddings(chunks: List[TextChunk], embeddings: List[List[float]]):
    """Store embeddings in vector database."""
    _, collection = get_vector_db()
    ids = [chunk.chunk_id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    metadatas = []
    chunk_index = 0
    while chunk_index < len(chunks):
        chunk = chunks[chunk_index]
        metadata = {
            "document_id": chunk.document_id,
            "language": chunk.language,
            "file_type": chunk.metadata.get("file_type", "unknown"),
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
        }
        metadatas.append(metadata)
        chunk_index += 1
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info(f"Stored {len(embeddings)} embeddings in vector database")


def generate_embeddings(documents: List[ProcessedDocument]) -> bool:
    """Main embedding agent: generates and stores embeddings for all documents."""
    logger.info(f"Generating embeddings for {len(documents)} documents")
    if not documents:
        logger.warning("No documents to process")
        return False
    all_chunks = []
    doc_index = 0
    while doc_index < len(documents):
        doc = documents[doc_index]
        all_chunks.extend(doc.chunks)
        doc_index += 1
    if not all_chunks:
        logger.warning("No chunks to generate embeddings for")
        return False
    try:
        embeddings = generate_embeddings_for_chunks(all_chunks)
        store_embeddings(all_chunks, embeddings)
        logger.info("Embeddings generated and stored successfully")
        return True
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return False

