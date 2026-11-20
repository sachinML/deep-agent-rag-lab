from evaluation import evaluate_faithfulness
import time
from litellm.exceptions import RateLimitError

CONTEXT = [
    (
        "[DOC-004] Hybrid Retrieval\n"
        "Hybrid retrieval combines lexical and dense retrieval "
        "results. Reciprocal Rank Fusion, also called RRF, "
        "can merge ranked lists using reciprocal rank "
        "contributions. Reranking can improve the ordering "
        "of candidate documents."
    )
]

CASES = [
    {
        "name": "Supported combination",
        "answer": (
            "Hybrid retrieval combines lexical and "
            "dense retrieval results."
        ),
        "expected_pass": True,
    },
    {
        "name": "Supported RRF",
        "answer": (
            "Reciprocal Rank Fusion can merge ranked lists."
        ),
        "expected_pass": True,
    },
    {
        "name": "Unsupported percentage",
        "answer": (
            "Hybrid retrieval improves accuracy by exactly "
            "35% compared with BM25."
        ),
        "expected_pass": False,
    },
    {
        "name": "Contradictory definition",
        "answer": (
            "Hybrid retrieval never uses lexical retrieval."
        ),
        "expected_pass": False,
    },
    {
        "name": "Mixed supported and unsupported",
        "answer": (
            "Hybrid retrieval combines lexical and dense "
            "retrieval using RRF. It improves accuracy "
            "by exactly 35% compared with BM25."
        ),
        "expected_pass": False,
    },
]


def evaluate_with_retry(case, context):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            return evaluate_faithfulness(
                question="Explain hybrid retrieval.",
                answer=case["answer"],
                retrieval_context=context,
            )

        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            delay = min(15 * (2 ** attempt), 60)

            print(
                f"[RATE LIMIT] "
                f"Retrying after {delay}s"
            )

            time.sleep(delay)


correct = 0

for case in CASES:
    result = evaluate_with_retry(case, CONTEXT)

    predicted = result["passed"]
    expected = case["expected_pass"]

    matched = predicted == expected
    correct += int(matched)

    print(f"\nCase: {case['name']}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Score: {result['score']}")
    print(f"Feedback: {result['feedback']}")

    # Space out evaluations on the free tier
    time.sleep(10)

print(f"\nAgreement: {correct}/{len(CASES)}")
