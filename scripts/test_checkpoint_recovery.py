import sys
from uuid import uuid4

from langgraph.types import Command

from main import agent


def start():
    thread_id = f"recovery-{uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Create a support ticket because "
                        "RAG retrieval returns irrelevant documents."
                    ),
                }
            ]
        },
        config=config,
    )

    if not result.get("__interrupt__"):
        print("ERROR: No approval interrupt occurred.")
        return

    print("\n=== WORKFLOW PAUSED ===")
    print("Thread ID:", thread_id)

    for item in result["__interrupt__"]:
        print("Pending approval:", item.value)

    print("\nExit this process now.")
    print("Resume using:")
    print(
        f"uv run python -m scripts.test_checkpoint_recovery "
        f"resume {thread_id}"
    )


def resume(thread_id: str):
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    snapshot = agent.get_state(config)

    if not snapshot.next:
        print("No interrupted workflow found for this thread.")
        return

    print("\n=== RECOVERED CHECKPOINT ===")
    print("Pending nodes:", snapshot.next)

    for task in snapshot.tasks:
        for item in task.interrupts:
            print("Pending approval:", item.value)

    decision = input("\nApprove ticket? (yes/no): ").strip().lower()

    if decision not in ("yes", "no"):
        print("Invalid decision. No action taken.")
        return

    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": (
                            "approve" if decision == "yes"
                            else "reject"
                        )
                    }
                ]
            }
        ),
        config=config,
    )

    print("\n=== RESUMED WORKFLOW ===")

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

    print("Remaining interrupts:", result.get("__interrupt__", []))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "start":
        start()
    elif len(sys.argv) == 3 and sys.argv[1] == "resume":
        resume(sys.argv[2])
    else:
        print("Usage:")
        print("  python -m scripts.test_checkpoint_recovery start")
        print("  python -m scripts.test_checkpoint_recovery resume <thread_id>")
