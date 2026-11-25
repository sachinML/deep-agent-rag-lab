import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

ES_URL = "http://localhost:9200"
INDEX_NAME = "rag_chunks_v1"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

QUESTIONS = [
    "How does RAG access external knowledge?",
    "What is non-parametric memory in RAG?",
    "How does the retriever improve question answering?",
    "What happens when the knowledge index is replaced?",
]

TOP_K = 5


def bm25_search(es, question, k=TOP_K):
    response = es.search(
        index=INDEX_NAME,
        size=k,
        query={
            "match": {
                "content": {
                    "query": question,
                }
            }
        },
        source=["chunk_id", "content"],
    )
    return response["hits"]["hits"]


def dense_search(es, model, question, k=TOP_K):
    query_vector = model.encode(
        QUERY_PREFIX + question, # model we are ysing require prefix to perform better
        normalize_embeddings=True,
    ).tolist()

    response = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": 30, # num_candidates is a per-shard setting. Each shard retrieves up to 30 candidates, and Elasticsearch merges results to return the global top k.
        },
        source=["chunk_id", "content"],
    )
    return response["hits"]["hits"]


def display_results(name, hits):
    print(f"\n{name}")

    for rank, hit in enumerate(hits, start=1):
        doc = hit["_source"]

        print(
            f"Rank {rank} | "
            f"Score {hit['_score']:.4f} | "
            f"Chunk {doc['chunk_id']}"
        )
        print(doc["content"][:200].replace("\n", " "))
        print()


def main():
    es = Elasticsearch(ES_URL)

    if not es.ping():
        raise ConnectionError("Elasticsearch is not running")

    model = SentenceTransformer(MODEL_NAME)

    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        bm25_hits = bm25_search(es, question)
        dense_hits = dense_search(es, model, question)

        display_results("BM25 RESULTS", bm25_hits)
        display_results("DENSE RESULTS", dense_hits)

        bm25_ids = {
            hit["_id"] for hit in bm25_hits
        }
        dense_ids = {
            hit["_id"] for hit in dense_hits
        }

        overlap = bm25_ids & dense_ids

        print(
            f"Top-{TOP_K} overlap: "
            f"{len(overlap)}/{TOP_K}"
        )
        print("Shared chunk IDs:", sorted(overlap))


if __name__ == "__main__":
    main()
