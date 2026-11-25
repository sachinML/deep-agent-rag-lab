
import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.deep_agent_rag.retrieval.compare import ES_URL, MODEL_NAME, QUESTIONS
from src.deep_agent_rag.retrieval.hybrid import hybrid_search


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATE_K = 20
FINAL_K = 5


def rerank_results(
    question: str,
    candidates: list[dict],
    reranker: CrossEncoder,
    k: int = FINAL_K,
) -> list[dict]:
    if not candidates:
        return []

    # Each candidate is compared directly with the question.
    pairs = [
        (question, hit["document"]["content"])
        for hit in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked = []

    for hit, score in zip(candidates, scores):
        ranked.append({
            **hit,
            "rerank_score": float(score),
        })

    ranked.sort(
        key=lambda item: (
            -item["rerank_score"],
            item["chunk_id"],
        )
    )

    return ranked[:k]


def hybrid_rerank_search(
    es: Elasticsearch,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    question: str,
    k: int = FINAL_K,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:

    # Retrieve candidates using hybrid RRF.
    candidates = hybrid_search(
        es,
        embedding_model,
        question,
        k=max(k, candidate_k),
    )

    # Rerank the hybrid candidates.
    return rerank_results(
        question=question,
        candidates=candidates,
        reranker=reranker,
        k=k,
    )


def main():
    es = Elasticsearch(ES_URL, request_timeout=60)

    if not es.ping():
        raise ConnectionError("Elasticsearch is unavailable")

    embedding_model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL)

    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        results = hybrid_rerank_search(
            es,
            embedding_model,
            reranker,
            question,
        )

        print("\nHYBRID + CROSS-ENCODER RESULTS")

        for rank, hit in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk: {hit['chunk_id']}")
            print(f"RRF score: {hit['score']:.6f}")
            print(f"Reranker score: {hit['rerank_score']:.4f}")
            print(f"Source ranks: {hit['ranks']}")
            print(
                "Content:",
                hit["document"]["content"][:250].replace("\n", " "),
            )


if __name__ == "__main__":
    main()
