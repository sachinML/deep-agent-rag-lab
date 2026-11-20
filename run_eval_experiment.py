
import argparse
from pathlib import Path
from uuid import uuid4
import re
import time

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from langfuse import Evaluation, get_client
from langfuse.langchain import CallbackHandler

# Reuse the existing application.
from main import workflow

langfuse = get_client()
langfuse_handler = CallbackHandler()

DATASET_NAME = "deep-agent-rag-baseline"


def invoke_with_retry(payload, config, max_retries=4):
    for attempt in range(max_retries):
        try:
            return workflow.invoke(payload, config=config)

        except Exception as exc:
            message = str(exc)

            is_rate_limit = (
                "RESOURCE_EXHAUSTED" in message
                or "429" in message
            )

            if not is_rate_limit:
                raise

            if attempt == max_retries - 1:
                raise

            # Respect Google's suggested retry delay.
            match = re.search(
                r"retry in ([\d.]+)s",
                message,
                flags=re.IGNORECASE,
            )

            if match:
                delay = float(match.group(1)) + 2
            else:
                delay = min(10 * (2 ** attempt), 90)

            print(
                f"[RATE LIMIT] Waiting {delay:.1f}s "
                f"before retry {attempt + 2}/{max_retries}"
            )
            time.sleep(delay)

    raise RuntimeError("Retries exhausted")


def run_agent(*, item, **kwargs):
    question = item.input["question"]

    print(f"\n[EXPERIMENT] Question: {question}")

    result = invoke_with_retry(
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

    time.sleep(20)

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
        return Evaluation(
            name="faithfulness",
            value=0.0,
            comment="No faithfulness score returned.",
        )

    return Evaluation(
        name="faithfulness",
        value=float(score),
        comment=output.get("feedback", ""),
    )


def pass_evaluator(*, output, **kwargs):
    return Evaluation(
        name="evaluation_passed",
        value=bool(output.get("passed", False)),
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
