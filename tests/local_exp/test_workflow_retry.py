
from types import SimpleNamespace
from unittest.mock import patch

import src.deep_agent_rag.orchestration.workflow as workflow


QUESTION = "Compare BM25 and dense retrieval."

CONTEXT = [
    "[DOC-002] BM25 ranks documents using lexical matching.",
    "[DOC-003] Dense retrieval uses embeddings for semantic search.",
]


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
            # Simulate initial retrieval.
            self.collector.context = list(CONTEXT)
            self.collector.documents = ["DOC-002", "DOC-003"]

            answer = (
                "BM25 uses embeddings [DOC-002]. "
                "Dense retrieval uses vectors [DOC-003]."
            )

        else:
            # Simulate generation-only retry:
            # No new documents are retrieved.
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

    call_counts = {
        "relevancy": 0,
        "sufficiency": 0,
        "faithfulness": 0,
    }

    def fake_relevancy(question, retrieval_context):
        call_counts["relevancy"] += 1
        return {
            "relevant": True,
            "score": 1.0,
            "feedback": "Relevant evidence.",
        }

    def fake_sufficiency(question, retrieval_context):
        call_counts["sufficiency"] += 1
        return {
            "sufficient": True,
            "score": 1.0,
            "feedback": "Evidence is sufficient.",
        }

    def fake_faithfulness(question, answer, retrieval_context):
        call_counts["faithfulness"] += 1

        if call_counts["faithfulness"] == 1:
            return {
                "passed": False,
                "score": 0.5,
                "feedback": "BM25 does not use embeddings.",
            }

        return {
            "passed": True,
            "score": 1.0,
            "feedback": "The corrected answer is grounded.",
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
                "thread_id": "retry-test",
                "evaluation": {},
                "attempts": 0,
                "retrieval_context": [],
            }
        )

    print("\n=== TEST RESULTS ===")
    print("Attempts:", result["attempts"])
    print("Failure type:", result["failure_type"])
    print("Agent invocations:", len(agent.calls))
    print("Relevancy evaluations:", call_counts["relevancy"])
    print("Sufficiency evaluations:", call_counts["sufficiency"])
    print("Faithfulness evaluations:", call_counts["faithfulness"])

    assert result["attempts"] == 2
    assert result["failure_type"] == "PASSED"
    assert len(agent.calls) == 2
    assert call_counts["relevancy"] == 1
    assert call_counts["sufficiency"] == 1
    assert call_counts["faithfulness"] == 2
    assert result["retrieval_context"] == CONTEXT

    assert "Reuse the following evidence" in agent.calls[1]

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
