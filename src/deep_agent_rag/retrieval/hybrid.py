
import os

# Avoid sending expired cached credentials for public models.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from collections import defaultdict

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from src.deep_agent_rag.retrieval.compare import (
    ES_URL,
    INDEX_NAME,
    MODEL_NAME,
    bm25_search,
    dense_search,
    QUESTIONS,
)

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


def reciprocal_rank_fusion(
    bm25_hits: list[dict],
    dense_hits: list[dict],
    rank_constant: int = RRF_K,
) -> list[dict]:
    scores = defaultdict(float)
    documents = {}
    ranks = defaultdict(dict)

    for retriever_name, hits in [
        ("bm25", bm25_hits),
        ("dense", dense_hits),
    ]:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["_id"]

            scores[chunk_id] += 1.0 / (
                rank_constant + rank
            )

            documents[chunk_id] = hit["_source"]
            ranks[chunk_id][retriever_name] = rank

    results = []

    for chunk_id, score in scores.items():
        results.append({
            "chunk_id": chunk_id,
            "score": score,
            "document": documents[chunk_id],
            "ranks": ranks[chunk_id],
        })

    return sorted(
        results,
        key=lambda item: (-item["score"], item["chunk_id"]),
    )


def hybrid_search(
    es: Elasticsearch,
    model: SentenceTransformer,
    question: str,
    k: int = TOP_K,
) -> list[dict]:
    candidate_k = max(CANDIDATE_K, k)

    bm25_hits = bm25_search(
        es,
        question,
        k=candidate_k,
    )

    dense_hits = dense_search(
        es,
        model,
        question,
        k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        bm25_hits,
        dense_hits,
    )

    return fused_results[:k]


def main():
    es = Elasticsearch(ES_URL)

    if not es.ping():
        raise ConnectionError("Elasticsearch is not running")

    model = SentenceTransformer(MODEL_NAME)

    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        results = hybrid_search(
            es,
            model,
            question,
        )

        print("\nHYBRID RRF RESULTS")

        for rank, result in enumerate(results, start=1):
            doc = result["document"]

            print(f"\nRank: {rank}")
            print(f"RRF Score: {result['score']:.6f}")
            print(f"Chunk: {result['chunk_id']}")
            print(f"Source ranks: {result['ranks']}")
            print(
                "Content:",
                doc["content"][:250].replace("\n", " "),
            )


if __name__ == "__main__":
    main()
