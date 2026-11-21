
from trajectory_evaluation import evaluate_trajectory


def test_efficient_agent():
    trajectory = [
        {"tool": "task", "status": "completed"},
        {"tool": "rag_search_chunks", "status": "completed"},
    ]

    result = evaluate_trajectory(
        trajectory=trajectory,
        required_tools=["task", "rag_search_chunks"],
        max_retrieval_calls=2,
    )

    assert result["passed"] is True
    assert result["retrieval_calls"] == 1


def test_excessive_retrieval():
    trajectory = [
        {"tool": "task", "status": "completed"},
        *[
            {"tool": "rag_search_chunks", "status": "completed"}
            for _ in range(4)
        ],
    ]

    result = evaluate_trajectory(
        trajectory=trajectory,
        required_tools=["task", "rag_search_chunks"],
        max_retrieval_calls=2,
    )

    assert result["passed"] is False
    assert result["retrieval_calls"] == 4
    assert result["within_retrieval_budget"] is False


def test_missing_delegation():
    trajectory = [
        {"tool": "rag_search_chunks", "status": "completed"},
    ]

    result = evaluate_trajectory(
        trajectory=trajectory,
        required_tools=["task", "rag_search_chunks"],
        max_retrieval_calls=2,
    )

    assert result["passed"] is False
    assert result["missing_tools"] == ["task"]
