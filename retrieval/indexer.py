# This script deletes and recreates only the development index rag_chunks_v1 whenever you rerun it.
# That's useful for experiments, but production pipelines should support incremental updates and deletions without rebuilding the index each time.

import json
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

ES_URL = "http://localhost:9200"
INDEX_NAME = "rag_chunks_v1"
CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def create_index(es: Elasticsearch):
    # Recreate the development index for a clean experiment.
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "section": {"type": "object"},
                        "chunk_index": {"type": "integer"},
                    }
                },
            }
        },
        settings={
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
    )

    print(f"[INDEX] Created {INDEX_NAME}")


def load_chunks() -> list[dict]:
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def index_chunks(es: Elasticsearch, chunks: list[dict]):
    actions = (
        {
            "_index": INDEX_NAME,
            "_id": chunk["chunk_id"],
            "_source": chunk,
        }
        for chunk in chunks
    )

    success, errors = helpers.bulk(
        es,
        actions,
        refresh="wait_for",
        raise_on_error=False,
    )

    print(f"[INDEX] Indexed: {success}")

    if errors:
        raise RuntimeError(
            f"Failed indexing {len(errors)} chunks: "
            f"{errors[:2]}"
        )


def main():
    es = Elasticsearch(ES_URL, request_timeout=60)

    if not es.ping():
        raise ConnectionError(
            "Elasticsearch is not available at localhost:9200"
        )

    chunks = load_chunks()

    print(f"[DATA] Loaded {len(chunks)} chunks")

    create_index(es)
    index_chunks(es, chunks)

    count = es.count(index=INDEX_NAME)["count"]

    print(f"[VERIFY] Elasticsearch document count: {count}")


if __name__ == "__main__":
    main()
