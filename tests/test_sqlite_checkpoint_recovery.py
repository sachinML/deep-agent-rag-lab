import sqlite3
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Generate a tool call only when no tool result exists."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(isinstance(message, ToolMessage) for message in messages):
            response = AIMessage(content="Ticket request completed.")
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_support_ticket",
                        "args": {
                            "title": "RAG retrieval issue",
                            "description": "Irrelevant documents returned",
                        },
                        "id": "call_ticket_recovery",
                        "type": "tool_call",
                    }
                ],
            )

        return ChatResult(
            generations=[ChatGeneration(message=response)]
        )


def build_test_agent(checkpointer, executed_tickets):
    @tool
    def create_support_ticket(title: str, description: str) -> str:
        """Create a support ticket."""
        executed_tickets.append(title)
        return f"Created: {title}"

    model = ToolCallingFakeModel(responses=[])

    return create_agent(
        model=model,
        tools=[create_support_ticket],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "create_support_ticket": {
                        "allowed_decisions": ["approve", "reject"]
                    }
                }
            )
        ],
        checkpointer=checkpointer,
    )


def test_sqlite_checkpoint_recovery(tmp_path):
    database_path = tmp_path / "checkpoints.sqlite"
    executed_tickets = []

    thread_id = f"recovery-test-{uuid4().hex}"
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # PROCESS A: Start the agent and pause before ticket creation.
    connection_a = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )

    try:
        checkpointer_a = SqliteSaver(connection_a)
        agent_a = build_test_agent(checkpointer_a, executed_tickets)

        result = agent_a.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a support ticket",
                    }
                ]
            },
            config=config,
        )

        assert result.get("__interrupt__")
        assert executed_tickets == []

    finally:
        connection_a.close()

    # PROCESS B (simulated): Reopen the saved database.
    connection_b = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )

    try:
        checkpointer_b = SqliteSaver(connection_b)
        agent_b = build_test_agent(checkpointer_b, executed_tickets)

        snapshot = agent_b.get_state(config)

        assert snapshot.next
        assert any(task.interrupts for task in snapshot.tasks)
        assert executed_tickets == []

        result = agent_b.invoke(
            Command(
                resume={
                    "decisions": [
                        {"type": "approve"}
                    ]
                }
            ),
            config=config,
        )

        assert executed_tickets == ["RAG retrieval issue"]
        assert not result.get("__interrupt__")
        assert not agent_b.get_state(config).next

    finally:
        connection_b.close()
