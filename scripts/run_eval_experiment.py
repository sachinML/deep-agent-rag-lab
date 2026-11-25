
import argparse
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langfuse import Evaluation, get_client
from langfuse.langchain import CallbackHandler

# Reuse the existing application.
from main import workflow

langfuse = get_client()
langfuse_handler = CallbackHandler()

DATASET_NAME = "deep-agent-rag-baseline"


def run_agent(*, item, **kwargs):
    question = item.input["question"]

    print(f"\n[EXPERIMENT] Question: {question}")

    result = workflow.invoke(
        {
            "question": question,
            "answer": "",
            "thread_id": f"eval-{uuid4().hex}",
            "evaluation": {},
            "attempts": 0,
            "retrieval_context": [],
        },
        config={
            "callbacks": [langfuse_handler],
            "run_name": "agentic-rag-evaluation",
        },
    )

    evaluation = result["evaluation"]

    return {
        "answer": result["answer"],
        "faithfulness": evaluation.get("score"),
        "passed": evaluation.get("passed", False),
        "feedback": evaluation.get("feedback", ""),
        "attempts": result["attempts"],
    }


def faithfulness_evaluator(*, output, **kwargs):
    score = output.get("faithfulness")

    if score is None:
        print(
            "[EVALUATION] Faithfulness score unavailable. "
            "Skipping score recording."
        )
        return None

    return Evaluation(
        name="faithfulness",
        value=float(score),
        comment=output.get("feedback", ""),
    )



def pass_evaluator(*, output, **kwargs):
    if output.get("faithfulness") is None:
        return None

    return Evaluation(
        name="evaluation_passed",
        value=bool(output["passed"]),
        data_type="BOOLEAN",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    dataset = langfuse.get_dataset(DATASET_NAME)

    items = dataset.items

    if args.limit > 0:
        items = items[:args.limit]

    print(f"Running {len(items)} test cases")

    result = langfuse.run_experiment(
        name="deep-agent-rag-baseline-v1",
        description="Baseline agent and DeepEval faithfulness",
        data=items,
        task=run_agent,
        evaluators=[
            faithfulness_evaluator,
            pass_evaluator,
        ],
        max_concurrency=1,
    )

    print(result.format())
    langfuse.flush()


if __name__ == "__main__":
    main()
