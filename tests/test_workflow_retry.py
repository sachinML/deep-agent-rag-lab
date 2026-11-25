
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler

import src.deep_agent_rag.orchestration.workflow as workflow


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
        self.agent_retrieval_calls = 0

    def invoke(self, inputs, config=None):
        self.calls += 1

        prompt = inputs["messages"][0]["content"]
        collector = FakeCollector.latest

        if self.calls == 1:
            if self.scenario == "generation":
                collector.context = [
                    "[DOC-001] Non-parametric memory is external."
                ]
                answer = "Unsupported answer [DOC-001]"
            else:
                collector.context = [
                    "[DOC-001] Unrelated information."
                ]
                answer = "Initial answer [DOC-001]"

            collector.events = [
                {"tool": "rag_search_chunks"}
            ]
            self.agent_retrieval_calls += 1

        elif self.scenario == "generation":
            assert "Do not invoke the research agent" in prompt

            collector.context = []
            collector.events = []

            answer = "Supported answer [DOC-001]"

        elif self.scenario == "retrieval":
            assert "LangGraph has already performed" in prompt
            assert "[DOC-002]" in prompt

            collector.context = []
            collector.events = []

            answer = (
                "Non-parametric memory is external [DOC-002]"
            )

        elif self.scenario == "retrieval_ignored":
            # The agent does not call any retrieval tool.
            # LangGraph must already have recovered evidence.
            assert "LangGraph has already performed" in prompt
            assert "[DOC-002]" in prompt

            collector.context = []
            collector.events = []

            answer = (
                "Non-parametric memory is external [DOC-002]"
            )

        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}"
            )

        return {
            "messages": [
                SimpleNamespace(text=answer)
            ]
        }


class FakeRetrievalTool:
    def __init__(self):
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1

        assert "query" in inputs
        assert inputs["k"] == 8

        return [
            {
                "id": "DOC-002",
                "chunk_id": "DOC-002",
                "title": "RAG Memory",
                "content": "Non-parametric memory is external.",
            }
        ]


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
    retrieval_tool = FakeRetrievalTool()

    with (
        patch.object(
            workflow,
            "RequestEvidenceCollector",
            FakeCollector,
        ),
        patch.object(
            workflow,
            "rag_search_chunks",
            retrieval_tool,
        ),
        patch.object(
            workflow,
            "evaluate_retrieval",
            fake_retrieval,
        ),
        patch.object(
            workflow,
            "evaluate_sufficiency",
            fake_sufficiency,
        ),
        patch.object(
            workflow,
            "evaluate_faithfulness",
            fake_faithfulness,
        ),
    ):
        graph = workflow.build_workflow(
            agent,
            store=None,
        )

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
    assert agent.calls == 2
    assert result["failure_type"] == "PASSED"
    assert result["evaluation"]["passed"] is True
    assert len(result["attempt_history"]) == 2

    if scenario in ("retrieval", "retrieval_ignored"):
        # Initial retrieval happened inside the agent.
        assert agent.agent_retrieval_calls == 1

        # The recovery node must explicitly call retrieval.
        assert retrieval_tool.calls == 1

        # The recovered passage must be accumulated.
        assert len(result["retrieval_context"]) == 2

        # Verify the recovery node was recorded.
        recovery_events = [
            event
            for event in result["trajectory"]
            if event.get("source") == "workflow_recovery"
        ]

        assert len(recovery_events) == 1
        assert recovery_events[0]["tool"] == "rag_search_chunks"

    elif scenario == "generation":
        # No additional retrieval during generation correction.
        assert agent.agent_retrieval_calls == 1
        assert retrieval_tool.calls == 0

        assert len(result["retrieval_context"]) == 1

    print(f"\nPASS: {scenario}")
    print("Agent calls:", agent.calls)
    print("Agent retrieval calls:", agent.agent_retrieval_calls)
    print("Forced retrieval calls:", retrieval_tool.calls)
    print("Final classification:", result["failure_type"])


if __name__ == "__main__":
    print("\n=== TEST 1: RETRIEVAL FAILURE ===")
    run_scenario("retrieval")

    print("\n=== TEST 2: GENERATION FAILURE ===")
    run_scenario("generation")

    print("\n=== TEST 3: AGENT IGNORES RETRIEVAL ===")
    run_scenario("retrieval_ignored")

    print("\nPASS: All deterministic recovery tests")
