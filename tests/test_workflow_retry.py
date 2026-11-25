
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler

import workflow


class FakeCollector(BaseCallbackHandler):
    latest = None

    def __init__(self):
        super().__init__()
        self.context = []
        self.events = []
        FakeCollector.latest = self

    def get_context(self):
        return self.context

    def get_documents(self):
        return self.context

    def get_trajectory(self):
        return self.events


class FakeAgent:
    def __init__(self, scenario):
        self.scenario = scenario
        self.calls = 0
        self.retrieval_calls = 0

    def invoke(self, inputs, config=None):
        self.calls += 1
        prompt = inputs["messages"][0]["content"]
        collector = FakeCollector.latest

        if self.scenario == "retrieval":
            if self.calls == 1:
                collector.context = [
                    "[DOC-001] Unrelated information."
                ]
                answer = "Initial answer [DOC-001]"
            else:
                assert "Delegate to the research agent" in prompt
                collector.context = [
                    "[DOC-002] Non-parametric memory is external."
                ]
                answer = "Non-parametric memory is external [DOC-002]"

            self.retrieval_calls += 1
            collector.events = [{"tool": "rag_search_chunks"}]

        else:
            if self.calls == 1:
                collector.context = [
                    "[DOC-001] Non-parametric memory is external."
                ]
                answer = "Unsupported answer [DOC-001]"
                self.retrieval_calls += 1
                collector.events = [{"tool": "rag_search_chunks"}]
            else:
                assert "Do not invoke the research agent" in prompt
                collector.context = []
                answer = "Supported answer [DOC-001]"

        return {
            "messages": [
                SimpleNamespace(text=answer)
            ]
        }


def fake_retrieval(question, retrieval_context):
    return {
        "relevant": True,
        "score": 1.0,
        "feedback": "Mock retrieval evaluation",
    }


def fake_sufficiency(question, retrieval_context):
    sufficient = any(
        "Non-parametric memory is external" in passage
        for passage in retrieval_context
    )
    return {
        "sufficient": sufficient,
        "score": float(sufficient),
        "feedback": "Mock sufficiency evaluation",
    }


def fake_faithfulness(question, answer, retrieval_context):
    passed = not answer.startswith("Unsupported")
    return {
        "passed": passed,
        "score": float(passed),
        "feedback": "Mock faithfulness evaluation",
    }


def run_scenario(scenario):
    agent = FakeAgent(scenario)

    with (
        patch.object(workflow, "RequestEvidenceCollector", FakeCollector),
        patch.object(workflow, "evaluate_retrieval", fake_retrieval),
        patch.object(workflow, "evaluate_sufficiency", fake_sufficiency),
        patch.object(workflow, "evaluate_faithfulness", fake_faithfulness),
    ):
        graph = workflow.build_workflow(agent, store=None)

        result = graph.invoke(
            {
                "question": "What is non-parametric memory?",
                "thread_id": f"test-{scenario}",
                "answer": "",
                "attempts": 0,
                "evaluation": {},
                "retrieval_context": [],
            },
            config={
                "configurable": {
                    "thread_id": f"test-{scenario}"
                }
            },
        )

    assert result["attempts"] == 2
    assert result["failure_type"] == "PASSED"
    assert result["evaluation"]["passed"] is True

    if scenario == "retrieval":
        assert agent.retrieval_calls == 2
        assert len(result["retrieval_context"]) == 2
    else:
        assert agent.retrieval_calls == 1
        assert len(result["retrieval_context"]) == 1

    print(f"\nPASS: {scenario} scenario")
    print("Generation attempts:", agent.calls)
    print("Simulated retrieval calls:", agent.retrieval_calls)
    print("Final classification:", result["failure_type"])


if __name__ == "__main__":
    run_scenario("retrieval")
    run_scenario("generation")
    print("\nPASS: Both workflow retry scenarios")
