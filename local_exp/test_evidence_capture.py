
import json
from uuid import uuid4
from types import SimpleNamespace

from evidence_capture import RequestEvidenceCollector


def main():
    collector_a = RequestEvidenceCollector()
    collector_b = RequestEvidenceCollector()

    run_a = uuid4()
    run_b = uuid4()

    collector_a.on_tool_start(
        {"name": "rag_search_chunks"},
        "{}",
        run_id=run_a,
    )

    collector_b.on_tool_start(
        {"name": "rag_search_chunks"},
        "{}",
        run_id=run_b,
    )

    collector_a.on_tool_end(
        SimpleNamespace(
            content=json.dumps([
                {
                    "id": "DOC-002",
                    "title": "BM25 Retrieval",
                    "content": "BM25 uses lexical matching.",
                }
            ])
        ),
        run_id=run_a,
    )

    collector_b.on_tool_end(
        SimpleNamespace(
            content=json.dumps([
                {
                    "id": "DOC-005",
                    "title": "RAGAS",
                    "content": "RAGAS evaluates RAG systems.",
                }
            ])
        ),
        run_id=run_b,
    )

    docs_a = collector_a.get_documents()
    docs_b = collector_b.get_documents()

    print("Request A:", docs_a)
    print("Request B:", docs_b)

    assert len(docs_a) == 1
    assert len(docs_b) == 1

    assert docs_a[0]["id"] == "DOC-002"
    assert docs_b[0]["id"] == "DOC-005"

    assert "DOC-005" not in collector_a.get_context()[0]
    assert "DOC-002" not in collector_b.get_context()[0]

    print("\nALL EVIDENCE ISOLATION TESTS PASSED")


if __name__ == "__main__":
    main()
