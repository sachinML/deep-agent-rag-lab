from collections import Counter


def evaluate_trajectory(
    trajectory: list[dict],
    required_tools: list[str],
    max_retrieval_calls: int | None = None,
) -> dict:
    """Evaluate observable tool-use behavior."""

    completed = [
        event
        for event in trajectory
        if event["status"] == "completed"
    ]

    failed = [
        event
        for event in trajectory
        if event["status"] == "failed"
    ]

    tool_counts = Counter(
        event["tool"] for event in completed
    )

    missing_tools = [
        tool
        for tool in required_tools
        if tool_counts[tool] == 0
    ]

    retrieval_calls = tool_counts["rag_search_chunks"]

    within_budget = (
        max_retrieval_calls is None
        or retrieval_calls <= max_retrieval_calls
    )

    return {
        "passed": (
            not missing_tools
            and not failed
            and within_budget
        ),
        "missing_tools": missing_tools,
        "failed_tool_calls": len(failed),
        "retrieval_calls": retrieval_calls,
        "within_retrieval_budget": within_budget,
        "tool_counts": dict(tool_counts),
    }
