
from evaluation import evaluate_faithfulness

cases = [
    (
        "Supported paraphrase",
        "The retriever searches an external document collection.",
        "The system finds relevant information in an external knowledge base.",
        True,
    ),
    (
        "Unsupported fusion method",
        "The system uses BM25 and dense vector search.",
        "The system combines BM25 and dense retrieval using RRF.",
        False,
    ),
    (
        "Supported numerical claim",
        "The system improved recall from 70% to 85%.",
        "Retrieval recall increased by 15 percentage points.",
        True,
    ),
    (
        "Unsupported numerical claim",
        "The evaluation showed improved retrieval performance.",
        "Retrieval precision improved by 20%.",
        False,
    ),
    (
        "Supported summary",
        "The retriever uses cosine similarity to rank document embeddings.",
        "Documents are ranked using cosine similarity.",
        True,
    ),
    (
        "Unsupported architecture",
        "The application uses a vector database for retrieval.",
        "The application uses Elasticsearch with HNSW indexing.",
        False,
    ),
]


def main():
    correct = 0

    for name, context, answer, expected in cases:
        result = evaluate_faithfulness(
            question="Evaluate the technical claim.",
            answer=answer,
            retrieval_context=[context],
        )

        predicted = result["passed"]
        match = predicted == expected
        correct += int(match)

        print(f"\n{name}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Score: {result['score']}")
        print(f"Match: {match}")
        print(f"Reason: {result['feedback']}")

    print(f"\nFINAL AGREEMENT: {correct}/{len(cases)}")


if __name__ == "__main__":
    main()
