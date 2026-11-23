
from evaluation import evaluate_faithfulness
from retrieval_evaluation import evaluate_sufficiency


TEST_CASES = [
    {
        "id": "T01",
        "question": "What is non-parametric memory in RAG?",
        "context": [
            "RAG combines parametric memory in a pretrained "
            "generator with non-parametric memory stored in "
            "an external document index. The retriever "
            "accesses the index to obtain relevant passages."
        ],
        "answer": (
            "Non-parametric memory is externally stored "
            "knowledge accessed through a retriever."
        ),
        "expected_faithfulness": True,
        "expected_sufficiency": True,
    },
    {
        "id": "T02",
        "question": "What is non-parametric memory in RAG?",
        "context": [
            "RAG combines parametric memory in a pretrained "
            "generator with non-parametric memory stored in "
            "an external document index."
        ],
        "answer": (
            "Non-parametric memory is stored entirely "
            "inside the generator's model weights."
        ),
        "expected_faithfulness": False,
        "expected_sufficiency": True,
    },
    {
        "id": "T03",
        "question": "Which retrieval algorithm does this RAG "
                    "system use?",
        "context": [
            "The system retrieves relevant passages "
            "from an external document index."
        ],
        "answer": "The system uses BM25.",
        "expected_faithfulness": False,
        "expected_sufficiency": False,
    },
    {
        "id": "T04",
        "question": "Which retrieval algorithm does this RAG "
                    "system use?",
        "context": [
            "The retriever uses BM25, a lexical ranking "
            "algorithm based on term matching."
        ],
        "answer": "The system uses BM25.",
        "expected_faithfulness": True,
        "expected_sufficiency": True,
    },
]


def main():
    correct_faithfulness = 0
    correct_sufficiency = 0

    for case in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"TEST: {case['id']}")
        print(f"QUESTION: {case['question']}")
        print(f"ANSWER: {case['answer']}")

        faithfulness = evaluate_faithfulness(
            question=case["question"],
            answer=case["answer"],
            retrieval_context=case["context"],
        )

        sufficiency = evaluate_sufficiency(
            question=case["question"],
            retrieval_context=case["context"],
        )

        faithfulness_match = (
            faithfulness["passed"]
            == case["expected_faithfulness"]
        )
        sufficiency_match = (
            sufficiency["sufficient"]
            == case["expected_sufficiency"]
        )

        correct_faithfulness += int(faithfulness_match)
        correct_sufficiency += int(sufficiency_match)

        print("\nFAITHFULNESS")
        print("Expected:", case["expected_faithfulness"])
        print("Predicted:", faithfulness["passed"])
        print("Score:", faithfulness["score"])
        print("Match:", faithfulness_match)
        print("Reason:", faithfulness["feedback"])

        print("\nSUFFICIENCY")
        print("Expected:", case["expected_sufficiency"])
        print("Predicted:", sufficiency["sufficient"])
        print("Match:", sufficiency_match)
        print("Reason:", sufficiency["feedback"])

    total = len(TEST_CASES)

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(
        f"Faithfulness agreement: "
        f"{correct_faithfulness}/{total}"
    )
    print(
        f"Sufficiency agreement: "
        f"{correct_sufficiency}/{total}"
    )


if __name__ == "__main__":
    main()
