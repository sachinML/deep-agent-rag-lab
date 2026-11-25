# This script generates document embeddings locally, then updates existing Elasticsearch records by their chunk_id.
# It also checks that the number of JSONL chunks matches the index count and that the corresponding IDs exist.
# For a production pipeline, we'd add content hashes and embedding-model version metadata to detect changed documents.
# Here, we're building a reproducible local baseline.

import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from pathlib import Path
import json

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

ES_URL = "http://localhost:9200"
INDEX_NAME = "rag_chunks_v1"
CHUNKS_FILE = Path("data/processed/chunks.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_FIELD = "embedding"
VECTOR_DIMS = 384


def add_vector_mapping(es: Elasticsearch) -> None:
    mapping = es.indices.get_mapping(index=INDEX_NAME)
    properties = mapping[INDEX_NAME]["mappings"].get(
        "properties", {}
    )

    if VECTOR_FIELD in properties:
        existing_dims = properties[VECTOR_FIELD].get("dims")
        if existing_dims != VECTOR_DIMS:
            raise ValueError(
                f"Existing vector dimension is {existing_dims}; "
                f"expected {VECTOR_DIMS}"
            )
        print("[MAPPING] Vector field already exists")
        return

    es.indices.put_mapping(
        index=INDEX_NAME,
        properties={
            VECTOR_FIELD: {
                "type": "dense_vector",
                "dims": VECTOR_DIMS,
                "index": True,
                "similarity": "cosine",
            }
        },
    )

    print("[MAPPING] Added 384-dimensional vector field")


def load_chunks() -> list[dict]:
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main() -> None:
    es = Elasticsearch(ES_URL, request_timeout=120)

    if not es.ping():
        raise ConnectionError("Elasticsearch is not running")

    if not es.indices.exists(index=INDEX_NAME):
        raise RuntimeError(
            f"Index {INDEX_NAME} does not exist. "
            "Run retrieval.indexer first."
        )

    chunks = load_chunks()
    print(f"[DATA] Loaded {len(chunks)} chunks")

    # Ensure every local chunk exists in Elasticsearch.
    indexed_count = es.count(index=INDEX_NAME)["count"]
    if indexed_count != len(chunks):
        raise ValueError(
            f"Index contains {indexed_count} documents, "
            f"but JSONL contains {len(chunks)}. "
            "Synchronize the documents before embedding."
        )

    missing_ids = [
        chunk["chunk_id"]
        for chunk in chunks
        if not es.exists(
            index=INDEX_NAME,
            id=chunk["chunk_id"],
        )
    ]

    if missing_ids:
        raise ValueError(
            f"Missing chunk IDs in Elasticsearch: {missing_ids[:5]}"
        )

    model = SentenceTransformer(MODEL_NAME)
    dimension = model.get_sentence_embedding_dimension()

    if dimension != VECTOR_DIMS:
        raise ValueError(
            f"Expected {VECTOR_DIMS} dimensions, got {dimension}"
        )

    add_vector_mapping(es)

    texts = [chunk["content"] for chunk in chunks]

    print("[EMBEDDING] Generating document embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    actions = (
        {
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": chunk["chunk_id"],
            "doc": {
                VECTOR_FIELD: vector.tolist()
            },
        }
        for chunk, vector in zip(chunks, embeddings)
    )

    success, errors = helpers.bulk(
        es,
        actions,
        refresh="wait_for",
        raise_on_error=False,
    )

    print(f"[INDEX] Updated vectors: {success}")

    if errors:
        raise RuntimeError(
            f"Failed to update {len(errors)} documents: "
            f"{errors[:2]}"
        )

    print("[DONE] Dense embeddings stored in Elasticsearch")


if __name__ == "__main__":
    main()
