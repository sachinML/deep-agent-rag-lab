
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client

# Load Langfuse credentials from .env
load_dotenv(Path(__file__).resolve().parent / ".env")

langfuse = get_client()

DATASET_NAME = "deep-agent-rag-baseline"

# Each test case contains:
# - id: unique test case identifier
# - question: input to our RAG agent
# - expected: criteria for a correct answer
# - category: type of capability being tested

test_cases = [
    {
        "id": "T01",
        "question": "What is BM25?",
        "expected": (
            "Explain lexical retrieval, term frequency, "
            "inverse document frequency, and document length "
            "normalization. Cite DOC-002."
        ),
        "category": "basic_retrieval",
    },
    {
        "id": "T02",
        "question": "What is dense retrieval?",
        "expected": (
            "Explain embeddings and vector similarity search. "
            "Mention dependence on the embedding model and domain. "
            "Cite DOC-003."
        ),
        "category": "basic_retrieval",
    },
    {
        "id": "T03",
        "question": "Explain hybrid retrieval and RRF.",
        "expected": (
            "Explain combining lexical and dense retrieval. "
            "Describe merging ranked lists using Reciprocal "
            "Rank Fusion. Cite DOC-004."
        ),
        "category": "multi_concept",
    },
    {
        "id": "T04",
        "question": "Compare BM25 and dense retrieval.",
        "expected": (
            "Compare keyword matching with semantic similarity. "
            "Describe relevant limitations. "
            "Cite DOC-002 and DOC-003."
        ),
        "category": "comparison",
    },
    {
        "id": "T05",
        "question": "What is reranking in a RAG pipeline?",
        "expected": (
            "Explain reordering retrieved candidates to improve "
            "their ranking. Cite DOC-007 or DOC-004."
        ),
        "category": "retrieval_detail",
    },
    {
        "id": "T06",
        "question": "What is RAGAS used for?",
        "expected": (
            "Explain RAG evaluation using information supported "
            "by DOC-005. Avoid unsupported metric details."
        ),
        "category": "evaluation_knowledge",
    },
    {
        "id": "T07",
        "question": "What is DeepEval used for?",
        "expected": (
            "Explain evaluation of LLM or RAG applications "
            "using DOC-006. Do not invent unsupported features."
        ),
        "category": "evaluation_knowledge",
    },
    {
        "id": "T08",
        "question": (
            "Compare BM25, dense retrieval, "
            "and hybrid retrieval."
        ),
        "expected": (
            "Compare all three approaches using DOC-002, "
            "DOC-003, and DOC-004. Avoid unsupported "
            "absolute performance claims."
        ),
        "category": "multi_document",
    },
    {
        "id": "T09",
        "question": (
            "Does hybrid retrieval always improve "
            "accuracy by exactly 35%?"
        ),
        "expected": (
            "Do not claim that hybrid retrieval always "
            "improves accuracy by 35%. Explain that the "
            "knowledge base does not establish this figure."
        ),
        "category": "unsupported_claim",
    },
    {
        "id": "T10",
        "question": (
            "Which GPU model runs our production "
            "retrieval service?"
        ),
        "expected": (
            "Acknowledge that the knowledge base does not "
            "specify the GPU model. Do not invent one."
        ),
        "category": "insufficient_evidence",
    },
]


def main():
    # 1. Create the dataset
    langfuse.create_dataset(
        name=DATASET_NAME,
        description=(
            "Evaluation dataset for Deep Agent RAG Lab. "
            "Tests retrieval, grounding, comparisons, "
            "and insufficient-evidence handling."
        ),
    )

    # 2. Upload dataset items
    for case in test_cases:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=case["id"],
            input={
                "question": case["question"],
            },
            expected_output={
                "criteria": case["expected"],
            },
            metadata={
                "case_id": case["id"],
                "category": case["category"],
            },
        )

        print(
            f"Uploaded {case['id']}: {case['question']}"
        )

    langfuse.flush()

    print(
        f"\nDataset ready: {DATASET_NAME}"
    )
    print(f"Total test cases: {len(test_cases)}")


if __name__ == "__main__":
    main()
