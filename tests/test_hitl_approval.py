from uuid import uuid4

import pytest

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Deterministic fake chat model supporting tool binding."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.fixture
def hitl_agent():
    executed_tickets = []

    @tool
    def create_support_ticket(title: str, description: str) -> str:
        """Create a support ticket."""
        executed_tickets.append(title)
        return f"Created: {title}"

    fake_model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_support_ticket",
                        "args": {
                            "title": "RAG retrieval issue",
                            "description": "Irrelevant documents returned",
                        },
                        "id": "call_ticket_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Request completed."),
        ]
    )

    agent = create_agent(
        model=fake_model,
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
        checkpointer=MemorySaver(),
    )

    return agent, executed_tickets


def start_request(agent):
    config = {
        "configurable": {
            "thread_id": f"test-{uuid4().hex}"
        }
    }

    result = agent.invoke(
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

    return result, config


def test_pending_approval_does_not_execute_tool(hitl_agent):
    agent, executed_tickets = hitl_agent

    result, _ = start_request(agent)

    assert result.get("__interrupt__")
    assert executed_tickets == []


def test_rejected_ticket_does_not_execute_tool(hitl_agent):
    agent, executed_tickets = hitl_agent

    result, config = start_request(agent)

    assert result.get("__interrupt__")

    resumed = agent.invoke(
        Command(
            resume={
                "decisions": [{"type": "reject"}]
            }
        ),
        config=config,
    )

    assert executed_tickets == []
    assert not resumed.get("__interrupt__")


def test_approved_ticket_executes_once(hitl_agent):
    agent, executed_tickets = hitl_agent

    result, config = start_request(agent)

    assert result.get("__interrupt__")
    assert executed_tickets == []

    resumed = agent.invoke(
        Command(
            resume={
                "decisions": [{"type": "approve"}]
            }
        ),
        config=config,
    )

    assert executed_tickets == ["RAG retrieval issue"]
    assert not resumed.get("__interrupt__")
