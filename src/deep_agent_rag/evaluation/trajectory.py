from collections import Counter


REQUIRED_TOOLS_BY_TYPE = {
    "rag": ["task", "rag_search_chunks"],
    "tool_status": ["get_system_status"],
}


def evaluate_trajectory(
    trajectory: list[dict],
    required_tools: list[str] | None = None,
    max_retrieval_calls: int | None = None,
    response_type: str = "rag",
) -> dict:
    """Evaluate tool execution according to the response type."""

    if required_tools is None:
        if response_type not in REQUIRED_TOOLS_BY_TYPE:
            raise ValueError(
                f"Unknown response type: {response_type}"
            )

        required_tools = REQUIRED_TOOLS_BY_TYPE[response_type]

    completed = [
        event
        for event in trajectory
        if event.get("status") == "completed"
    ]

    failed = [
        event
        for event in trajectory
        if event.get("status") == "failed"
    ]

    tool_counts = Counter(
        event["tool"]
        for event in completed
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
        "response_type": response_type,
        "missing_tools": missing_tools,
        "failed_tool_calls": len(failed),
        "retrieval_calls": retrieval_calls,
        "within_retrieval_budget": within_budget,
        "tool_counts": dict(tool_counts),
    }