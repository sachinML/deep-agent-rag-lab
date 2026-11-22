
import os

# Apply before importing Hugging Face libraries.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import json
import math
from pathlib import Path
from statistics import mean

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from retrieval.compare import (
    ES_URL,
    INDEX_NAME,
    MODEL_NAME,
    bm25_search,
    dense_search,
)
from retrieval.hybrid import hybrid_search

from sentence_transformers import CrossEncoder

from retrieval.reranker import (
    RERANKER_MODEL,
    hybrid_rerank_search,
)


DATASET_PATH = Path("data/evaluation/retrieval_eval.json")
TOP_K = 5


def load_dataset() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        raise ValueError("Evaluation dataset is empty")

    for item in dataset:
        if not item.get("relevant_chunk_ids"):
            raise ValueError(
                f"No relevance labels for {item['id']}"
            )

    return dataset


def extract_ids(hits: list[dict]) -> list[str]:
    return [hit["_id"] for hit in hits]


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    hits = len(set(retrieved[:k]) & relevant)
    return hits / k


def reciprocal_rank_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    # Binary relevance: relevant=1, irrelevant=0.
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(
            retrieved[:k], start=1
        )
        if chunk_id in relevant
    )

    ideal_count = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / idcg if idcg else 0.0


def calculate_metrics(
    retrieved: list[str],
    relevant: set[str],
    k: int = TOP_K,
) -> dict:
    return {
        "recall": recall_at_k(retrieved, relevant, k),
        "precision": precision_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank_at_k(retrieved, relevant, k),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }


def validate_labels(
    es: Elasticsearch,
    dataset: list[dict],
) -> None:
    all_ids = sorted({
        chunk_id
        for item in dataset
        for chunk_id in item["relevant_chunk_ids"]
    })

    response = es.mget(
        index=INDEX_NAME,
        ids=all_ids,
        source=False,
    )

    missing = [
        doc["_id"]
        for doc in response["docs"]
        if not doc.get("found", False)
    ]

    if missing:
        raise ValueError(
            "These labeled chunks do not exist in the index: "
            f"{missing}"
        )

    print(
        f"[VALIDATION] {len(all_ids)} unique labeled "
        "chunk IDs exist in Elasticsearch"
    )


def main() -> None:
    es = Elasticsearch(ES_URL, request_timeout=60)

    if not es.ping():
        raise ConnectionError(
            "Elasticsearch is unavailable"
        )

    dataset = load_dataset()
    validate_labels(es, dataset)

    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL)

    results = {
        "BM25": [],
        "Dense": [],
        "Hybrid": [],
        "Hybrid + Rerank": [],
    }

    for item in dataset:
        question = item["question"]
        relevant = set(item["relevant_chunk_ids"])

        bm25_hits = bm25_search(es, question, k=TOP_K)
        dense_hits = dense_search(
            es, model, question, k=TOP_K
        )
        hybrid_hits = hybrid_search(
            es, model, question, k=TOP_K
        )

        reranked_hits = hybrid_rerank_search(
            es,
            model,
            reranker,
            question,
            k=TOP_K,
            candidate_k=20,
        )

        retrieved_by_method = {
            "BM25": extract_ids(bm25_hits),
            "Dense": extract_ids(dense_hits),
            "Hybrid": [
                hit["chunk_id"] for hit in hybrid_hits
            ],
            "Hybrid + Rerank": [
                hit["chunk_id"] for hit in reranked_hits
            ],
        }

        print(f"\n{'=' * 65}")
        print(f"{item['id']}: {question}")
        print(f"Relevant chunks: {sorted(relevant)}")

        for method, retrieved in retrieved_by_method.items():
            metrics = calculate_metrics(
                retrieved, relevant, TOP_K
            )

            results[method].append(metrics)

            print(f"\n{method}")
            print(f"Retrieved: {retrieved}")
            print(
                f"Recall@{TOP_K}: "
                f"{metrics['recall']:.3f} | "
                f"Precision@{TOP_K}: "
                f"{metrics['precision']:.3f} | "
                f"RR@{TOP_K}: "
                f"{metrics['mrr']:.3f} | "
                f"NDCG@{TOP_K}: "
                f"{metrics['ndcg']:.3f}"
            )

    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 65)

    print(
        f"{'Method':<18}"
        f"{'Recall@5':>12}"
        f"{'Precision@5':>14}"
        f"{'MRR@5':>12}"
        f"{'NDCG@5':>12}"
    )

    for method, query_results in results.items():
        averages = {
            metric: mean(
                result[metric]
                for result in query_results
            )
            for metric in ("recall", "precision", "mrr", "ndcg")
        }

        print(
            f"{method:<10}"
            f"{averages['recall']:>12.3f}"
            f"{averages['precision']:>14.3f}"
            f"{averages['mrr']:>12.3f}"
            f"{averages['ndcg']:>12.3f}"
        )


if __name__ == "__main__":
    main()
