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


def main():
    es = Elasticsearch(ES_URL)
    model = SentenceTransformer(MODEL_NAME)

    question = "How does RAG access external knowledge?"

    query_vector = model.encode(
        QUERY_PREFIX + question,
        normalize_embeddings=True,
    ).tolist()

    response = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": 3,
            "num_candidates": 30,
        },
        source=["chunk_id", "content", "title"],
    )

    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        doc = hit["_source"]
        print(f"\nRank: {rank}")
        print(f"Score: {hit['_score']:.4f}")
        print(f"Chunk: {doc['chunk_id']}")
        print(f"Content: {doc['content'][:400]}")


if __name__ == "__main__":
    main()
