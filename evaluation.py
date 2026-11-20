import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import LiteLLMModel

# Load environment variables before initializing the judge
load_dotenv(
    Path(__file__).resolve().parent / ".env",
    override=True,
)

# Validate Groq credentials
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

judge_model = LiteLLMModel(
    model="groq/openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
)

from deepeval.metrics.faithfulness import FaithfulnessTemplate


class GroqFaithfulnessTemplate(FaithfulnessTemplate): # as groq is not follwing the deepeval template and showing error
    @staticmethod
    def generate_verdicts(*args, **kwargs):
        original_prompt = FaithfulnessTemplate.generate_verdicts(
            *args, **kwargs
        )

        return original_prompt + """

        IMPORTANT JSON OUTPUT REQUIREMENT:

        Every object in the verdicts array MUST include
        BOTH fields:
        - verdict
        - reason

        Include a reason for ALL verdicts, including "yes".

        Correct:
        {
            "verdicts": [
                {
                    "verdict": "yes",
                    "reason": "The claim is supported by the context."
                },
                {
                    "verdict": "borderline",
                    "reason": "The evidence is incomplete."
                }
            ]
        }

        Never omit the reason field.
        """


def evaluate_faithfulness(
    question: str,
    answer: str,
    retrieval_context: list[str],
) -> dict:

    if not retrieval_context:
        return {
            "passed": False,
            "score": 0.0,
            "feedback": "No retrieval evidence available.",
        }

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )

    metric = FaithfulnessMetric(
        threshold=0.8,
        model=judge_model,
        include_reason=True,
        async_mode=False,
        penalize_ambiguous_claims=True,
        verbose_mode=True,
        evaluation_template=GroqFaithfulnessTemplate,
    )

    try:
        metric.measure(test_case)

    except Exception as error:
        print(
            f"\n[DEEPEVAL ERROR] "
            f"{type(error).__name__}: {error}"
        )
        raise

    return {
        "passed": bool(metric.is_successful()),
        "score": metric.score,
        "feedback": metric.reason or "No explanation returned.",
    }
