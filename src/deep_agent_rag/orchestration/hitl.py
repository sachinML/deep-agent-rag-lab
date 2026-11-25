
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class TicketState(TypedDict):
    title: str
    description: str
    approved: bool | None
    status: str

# When execution reaches interrupt(), LangGraph pauses the workflow and saves its state using the checkpointer.
def request_approval(state: TicketState):
    decision = interrupt(
        {
            "action": "create_support_ticket",
            "title": state["title"],
            "description": state["description"],
            "message": "Approve support ticket creation?",
        }
    )

    return {"approved": decision == "approve"}


def route_approval(
    state: TicketState,
) -> Literal["create_ticket", "reject_ticket"]:
    if state["approved"]:
        return "create_ticket"
    return "reject_ticket"


def create_ticket(state: TicketState):
    # Mock action: no external ticket is created.
    print(f"[MOCK] Creating ticket: {state['title']}")
    return {"status": "created"}


def reject_ticket(state: TicketState):
    return {"status": "rejected"}


def build_hitl_workflow():
    builder = StateGraph(TicketState)

    builder.add_node("request_approval", request_approval)
    builder.add_node("create_ticket", create_ticket)
    builder.add_node("reject_ticket", reject_ticket)

    builder.add_edge(START, "request_approval")

    builder.add_conditional_edges(
        "request_approval",
        route_approval,
    )

    builder.add_edge("create_ticket", END)
    builder.add_edge("reject_ticket", END)

    return builder.compile(checkpointer=MemorySaver())
