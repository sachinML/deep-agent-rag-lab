
import pytest

from knowledge import (
    load_documents,
    search_documents,
    tokenize,
)


def test_documents_loaded():
    documents = load_documents()

    assert len(documents) == 7
    assert documents[0]["id"] == "DOC-001"


def test_tokenization():
    tokens = tokenize("What is RAGAS faithfulness?")

    assert "ragas" in tokens
    assert "faithfulness" in tokens
    assert "what" not in tokens


def test_ragas_retrieval():
    results = search_documents(
        "RAGAS faithfulness",
        k=3,
    )

    assert len(results) > 0
    assert results[0]["id"] == "DOC-005"


def test_bm25_retrieval():
    results = search_documents(
        "BM25 lexical retrieval",
        k=3,
    )

    assert results[0]["id"] == "DOC-002"


def test_top_k_limit():
    results = search_documents("retrieval", k=2)

    assert len(results) <= 2


def test_empty_query():
    assert search_documents("") == []


def test_invalid_k():
    with pytest.raises(ValueError):
        search_documents("RAGAS", k=0)


def test_no_results():
    results = search_documents("quantum teleportation")

    assert results == []
