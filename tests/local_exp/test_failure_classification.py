from src.deep_agent_rag.orchestration.workflow import classify_failure

test_cases = [
    {
        "name": "Successful answer",
        "retrieval_evaluation": {"sufficient": True},
        "evaluation": {"passed": True},
    },
    {
        "name": "Generation failure",
        "retrieval_evaluation": {"sufficient": True},
        "evaluation": {"passed": False},
    },
    {
        "name": "Retrieval failure",
        "retrieval_evaluation": {"sufficient": False},
        "evaluation": {"passed": False},
    },
]

for case in test_cases:
    result = classify_failure(case)
    print(f"{case['name']}: {result}")
