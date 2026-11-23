
import os

# Must be set before importing Hugging Face libraries.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import csv
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder

from retrieval.compare import (
    ES_URL,
    INDEX_NAME,
    MODEL_NAME,
    bm25_search,
    dense_search,
)
from retrieval.hybrid import hybrid_search
from retrieval.reranker import (
    RERANKER_MODEL,
    hybrid_rerank_search,
)

DATASET_FILE = Path("data/evaluation/retrieval_eval.json")
OUTPUT_FILE = Path("data/evaluation/relevance_review.csv")

TOP_K = 10


def load_questions():
    with DATASET_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def get_chunk_ids(hits, method):
    if method in {"BM25", "Dense"}:
        return [hit["_id"] for hit in hits]

    return [hit["chunk_id"] for hit in hits]


def build_review_rows(
    es,
    embedding_model,
    reranker,
    dataset,
):
    rows = []

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        existing_labels = set(item["relevant_chunk_ids"])

        print(f"[REVIEW] Processing {question_id}: {question}")

        search_results = {
            "BM25": bm25_search(
                es, question, k=TOP_K
            ),
            "Dense": dense_search(
                es, embedding_model, question, k=TOP_K
            ),
            "Hybrid": hybrid_search(
                es, embedding_model, question, k=TOP_K
            ),
            "Hybrid + Rerank": hybrid_rerank_search(
                es,
                embedding_model,
                reranker,
                question,
                k=TOP_K,
                candidate_k=20,
            ),
        }

        ranks = {}
        candidate_ids = set(existing_labels)

        for method, hits in search_results.items():
            chunk_ids = get_chunk_ids(hits, method)

            for rank, chunk_id in enumerate(
                chunk_ids, start=1
            ):
                candidate_ids.add(chunk_id)
                ranks.setdefault(chunk_id, {})[method] = rank

        response = es.mget(
            index=INDEX_NAME,
            ids=sorted(candidate_ids),
        )

        documents = {
            doc["_id"]: doc["_source"]
            for doc in response["docs"]
            if doc.get("found")
        }

        missing = candidate_ids - set(documents)
        if missing:
            raise ValueError(
                f"Missing indexed chunks: {sorted(missing)}"
            )

        for chunk_id in sorted(candidate_ids):
            document = documents[chunk_id]
            method_ranks = ranks.get(chunk_id, {})
            metadata = document.get("metadata", {})

            rows.append({
                "question_id": question_id,
                "question": question,
                "chunk_id": chunk_id,
                "existing_gold_label": (
                    1 if chunk_id in existing_labels else 0
                ),
                "human_relevance": "",
                "bm25_rank": method_ranks.get("BM25", ""),
                "dense_rank": method_ranks.get("Dense", ""),
                "hybrid_rank": method_ranks.get("Hybrid", ""),
                "rerank_rank": method_ranks.get(
                    "Hybrid + Rerank", ""
                ),
                "source": metadata.get("source", ""),
                "section": json.dumps(
                    metadata.get("section", {}),
                    ensure_ascii=False,
                ),
                "content": document["content"],
            })

    return rows


def main():
    es = Elasticsearch(ES_URL, request_timeout=60)

    if not es.ping():
        raise ConnectionError("Elasticsearch is unavailable")

    dataset = load_questions()

    embedding_model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL)

    rows = build_review_rows(
        es,
        embedding_model,
        reranker,
        dataset,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "question",
                "chunk_id",
                "existing_gold_label",
                "human_relevance",
                "bm25_rank",
                "dense_rank",
                "hybrid_rank",
                "rerank_rank",
                "source",
                "section",
                "content",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Review file: {OUTPUT_FILE}")
    print(f"[DONE] Question-chunk pairs: {len(rows)}")


if __name__ == "__main__":
    main()
