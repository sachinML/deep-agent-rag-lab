
from src.deep_agent_rag.orchestration.workflow import classify_failure
from src.deep_agent_rag.orchestration.workflow import build_failure_retry_message

cases = [
    {
        "name": "Successful answer despite low context relevancy",
        "retrieval_evaluation": {
            "relevant": False,
            "sufficient": True,
        },
        "evaluation": {"passed": True},
        "expected": "PASSED",
    },
    {
        "name": "Insufficient retrieval",
        "retrieval_evaluation": {
            "relevant": True,
            "sufficient": False,
        },
        "evaluation": {"passed": False},
        "expected": "RETRIEVAL_FAILURE",
    },
    {
        "name": "Unfaithful answer with sufficient evidence",
        "retrieval_evaluation": {
            "relevant": True,
            "sufficient": True,
        },
        "evaluation": {"passed": False},
        "expected": "GENERATION_FAILURE",
    },
]

for case in cases:
    result = classify_failure(case)

    print(f"\n{case['name']}")
    print("Expected:", case["expected"])
    print("Actual:", result)

    assert result == case["expected"]

print("\nPASS: All failure classifications correct")

print("\n=== RETRY INSTRUCTION TEST ===")

base_state = {
    "question": "What is non-parametric memory?",
    "answer": "Previous answer",
    "retrieval_context": ["[DOC-001] Supporting evidence"],
    "evaluation": {
        "passed": False,
        "feedback": "Unsupported claim",
    },
    "retrieval_evaluation": {
        "sufficient": True,
        "sufficiency_feedback": "Enough evidence",
    },
}

# Generation failure: reuse evidence.
generation_message = build_failure_retry_message(base_state)

assert "Do not invoke the research agent" in generation_message
assert "EXISTING EVIDENCE" in generation_message

print("PASS: Generation retry reuses evidence")

# Retrieval failure: search for missing evidence.
retrieval_state = {
    **base_state,
    "retrieval_evaluation": {
        "sufficient": False,
        "sufficiency_feedback": "Missing definition",
    },
}

retrieval_message = build_failure_retry_message(retrieval_state)

assert "The previous attempt had insufficient evidence" in retrieval_message
assert "Missing definition" in retrieval_message

print("PASS: Retrieval retry requests new evidence")
