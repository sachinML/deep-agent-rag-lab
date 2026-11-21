
from types import SimpleNamespace
from unittest.mock import patch

import workflow


QUESTION = "Compare BM25 and dense retrieval."

BM25_CONTEXT = (
    "[DOC-002] BM25 uses lexical matching, "
    "term frequency, and inverse document frequency."
)

DENSE_CONTEXT = (
    "[DOC-003] Dense retrieval uses embeddings "
    "and vector similarity for semantic search."
)


class FakeEvidenceCollector:
    def __init__(self):
        self.context = []
        self.documents = []

    def clear(self):
        self.context = []
        self.documents = []

    def get_context(self):
        return list(self.context)

    def get_documents(self):
        return list(self.documents)


class FakeDeepAgent:
    def __init__(self, collector):
        self.collector = collector
        self.calls = []

    def invoke(self, inputs, config=None):
        prompt = inputs["messages"][-1]["content"]
        self.calls.append(prompt)

        if len(self.calls) == 1:
            # First attempt: retrieval returns only BM25.
            self.collector.context = [BM25_CONTEXT]
            self.collector.documents = ["DOC-002"]

            answer = (
                "BM25 uses lexical matching [DOC-002]. "
                "Dense retrieval information is unavailable."
            )

        else:
            # Second attempt: research finds missing dense evidence.
            self.collector.context = [DENSE_CONTEXT]
            self.collector.documents = ["DOC-003"]

            answer = (
                "BM25 uses lexical matching [DOC-002]. "
                "Dense retrieval uses embeddings [DOC-003]."
            )

        return {
            "messages": [SimpleNamespace(text=answer)]
        }


def main():
    collector = FakeEvidenceCollector()
    agent = FakeDeepAgent(collector)

    counts = {
        "relevancy": 0,
        "sufficiency": 0,
        "faithfulness": 0,
    }

    def fake_relevancy(question, retrieval_context):
        counts["relevancy"] += 1
        return {
            "relevant": True,
            "score": 1.0,
            "feedback": "Evidence is related to the question.",
        }

    def fake_sufficiency(question, retrieval_context):
        counts["sufficiency"] += 1

        has_bm25 = BM25_CONTEXT in retrieval_context
        has_dense = DENSE_CONTEXT in retrieval_context

        sufficient = has_bm25 and has_dense

        return {
            "sufficient": sufficient,
            "score": 1.0 if sufficient else 0.0,
            "feedback": (
                "Both retrieval methods are covered."
                if sufficient
                else "Missing information about dense retrieval."
            ),
        }

    def fake_faithfulness(question, answer, retrieval_context):
        counts["faithfulness"] += 1

        has_dense = DENSE_CONTEXT in retrieval_context

        return {
            "passed": has_dense,
            "score": 1.0 if has_dense else 0.5,
            "feedback": (
                "Answer is grounded."
                if has_dense
                else "The comparison is incomplete."
            ),
        }

    with (
        patch.object(workflow, "evidence_collector", collector),
        patch.object(workflow, "evaluate_retrieval", fake_relevancy),
        patch.object(workflow, "evaluate_sufficiency", fake_sufficiency),
        patch.object(workflow, "evaluate_faithfulness", fake_faithfulness),
    ):
        graph = workflow.build_workflow(agent, store=None)

        result = graph.invoke(
            {
                "question": QUESTION,
                "answer": "",
                "thread_id": "retrieval-failure-test",
                "evaluation": {},
                "attempts": 0,
                "retrieval_context": [],
            }
        )

    print("\n=== TEST RESULTS ===")
    print("Attempts:", result["attempts"])
    print("Failure type:", result["failure_type"])
    print("Agent invocations:", len(agent.calls))
    print("Relevancy evaluations:", counts["relevancy"])
    print("Sufficiency evaluations:", counts["sufficiency"])
    print("Faithfulness evaluations:", counts["faithfulness"])
    print("Final context:", result["retrieval_context"])

    assert result["attempts"] == 2
    assert result["failure_type"] == "PASSED"

    assert len(agent.calls) == 2

    assert counts["relevancy"] == 2
    assert counts["sufficiency"] == 2
    assert counts["faithfulness"] == 2

    assert result["retrieval_context"] == [
        BM25_CONTEXT,
        DENSE_CONTEXT,
    ]

    assert "Missing information about dense retrieval" in agent.calls[1]
    assert "Delegate to the research agent" in agent.calls[1]

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
