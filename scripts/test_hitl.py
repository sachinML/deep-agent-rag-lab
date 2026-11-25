
from uuid import uuid4

from langgraph.types import Command

from deep_agent_rag.orchestration.hitl import build_hitl_workflow


workflow = build_hitl_workflow()

config = {
    "configurable": {
        "thread_id": str(uuid4()),
    }
}

initial_state = {
    "title": "RAG retrieval returning irrelevant documents",
    "description": "Investigate low retrieval precision.",
    "approved": None,
    "status": "pending",
}

print("\n=== START WORKFLOW ===")

result = workflow.invoke(initial_state, config=config)

print("Result:", result)

snapshot = workflow.get_state(config)

print("Next nodes:", snapshot.next)

assert snapshot.next, "Expected workflow to be paused"

print("\n=== HUMAN APPROVAL ===")

decision = input("Approve ticket creation? (yes/no): ").strip().lower()

resume_value = "approve" if decision == "yes" else "reject"

result = workflow.invoke(
    Command(resume=resume_value),
    config=config,
)

print("\nFinal result:", result)
