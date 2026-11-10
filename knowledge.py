
import json
import re
from pathlib import Path

from langchain.tools import tool


DATA_PATH = Path(__file__).parent / "data" / "knowledge.json"

STOP_WORDS = {
    "the", "a", "an", "is", "are", "what", "how",
    "of", "to", "in", "for", "and", "or", "does",
    "with", "between", "explain", "compare"
}


def tokenize(text: str) -> set[str]:
    """Convert text into normalized search terms."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return set(words) - STOP_WORDS


def load_documents() -> list[dict]:
    """Load documents from our local knowledge base."""
    with DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def search_documents(query: str, k: int = 3) -> list[dict]:
    """
    Retrieve top-k documents based on keyword overlap.

    This is a deliberately simple retriever.
    It does not use embeddings or a vector database.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    query_terms = tokenize(query)

    if not query_terms:
        return []

    ranked = []

    for doc in load_documents():
        title_terms = tokenize(doc["title"])
        content_terms = tokenize(doc["content"])

        title_overlap = len(query_terms & title_terms)
        content_overlap = len(query_terms & content_terms)

        score = 3 * title_overlap + content_overlap

        if score > 0:
            ranked.append({
                **doc,
                "score": score,
            })

    ranked.sort(
        key=lambda item: (-item["score"], item["id"])
    )

    return ranked[:k]


@tool
def rag_search_chunks(query: str, k: int = 3) -> str:
    """
    Search the local RAG knowledge base.

    Use this tool when answering technical questions about
    RAG, retrieval, BM25, embeddings, reranking, RAGAS
    or DeepEval. Returns document IDs and relevant evidence.
    """
    results = search_documents(query=query, k=k)

    if not results:
        return "No relevant documents found."

    return json.dumps(results, indent=2)
