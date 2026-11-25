
import json
import os
from functools import lru_cache

# Public Hugging Face models: don't use expired implicit credentials.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from elasticsearch import Elasticsearch
from langchain.tools import tool
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.deep_agent_rag.retrieval.compare import ES_URL, MODEL_NAME, INDEX_NAME
from src.deep_agent_rag.retrieval.reranker import (
    RERANKER_MODEL,
    hybrid_rerank_search,
)


@lru_cache(maxsize=1)
def get_es_client() -> Elasticsearch:
    es = Elasticsearch(ES_URL, request_timeout=60)

    if not es.ping():
        raise ConnectionError(
            "Elasticsearch is unavailable at " + ES_URL
        )

    return es


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def search_documents(query: str, k: int = 3) -> list[dict]:
    """Search indexed chunks using hybrid retrieval and reranking."""

    if not query.strip():
        return []

    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")

    results = hybrid_rerank_search(
        es=get_es_client(),
        embedding_model=get_embedding_model(),
        reranker=get_reranker(),
        question=query,
        k=k,
        candidate_k=20,
    )

    documents = []

    for result in results:
        source = result["document"]
        metadata = source.get("metadata", {})

        chunk_id = source["chunk_id"]

        documents.append({
            # Required by our evidence collector.
            "id": chunk_id,
            "title": source.get("title", ""),
            "content": source["content"],

            # Chunk-level provenance.
            "chunk_id": chunk_id,
            "document_id": source.get("document_id"),
            "source": metadata.get("source", ""),
            "section": metadata.get("section", {}),

            # Diagnostic metadata.
            "rerank_score": result["rerank_score"],
            "rrf_score": result["score"],
        })

    return documents


@tool
def rag_search_chunks(query: str, k: int = 3) -> str:
    """
    Search the indexed research knowledge base using hybrid
    BM25 + dense retrieval followed by cross-encoder reranking.

    Use this tool for questions about concepts or experiments
    covered by the indexed documents.

    Returns relevant chunks with IDs suitable for citations.
    """
    results = search_documents(query=query, k=k)

    return json.dumps(results, ensure_ascii=False)
