
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from main import agent


def main():
    config = {
        "configurable": {
            "thread_id": f"hitl-test-{uuid4().hex}"
        },
        "recursion_limit": 60,
    }

    print("\n=== START DEEP AGENT ===")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Create a support ticket for an issue where "
                        "RAG retrieval is returning irrelevant documents."
                    )
                )
            ]
        },
        config=config,
    )

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        print("\nNo HITL interrupt occurred.")
        print("Result:", result)
        return

    print("\n=== HUMAN APPROVAL REQUIRED ===")

    for item in interrupts:
        print(item.value)

    decision = input(
        "\nApprove ticket creation? (yes/no): "
    ).strip().lower()

    if decision == "yes":
        human_decision = "approve"
    else:
        human_decision = "reject"

    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": human_decision}
                ]
            }
        ),
        config=config,
    )

    print("\n=== FINAL RESULT ===")

    for message in reversed(result.get("messages", [])):
        if message.type != "ai":
            continue

        content = message.content

        if isinstance(content, str):
            print(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block.get("text", ""))

        break

    print("\nInterrupts remaining:", result.get("__interrupt__", []))


if __name__ == "__main__":
    main()
