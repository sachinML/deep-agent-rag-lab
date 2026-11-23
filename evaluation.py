
import os
from pathlib import Path
from dotenv import load_dotenv

# Load project environment configuration.
load_dotenv(
    Path(__file__).resolve().parent / ".env",
    override=True,
)

from deepeval.models import OllamaModel

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics.faithfulness import FaithfulnessTemplate





# # Shared local evaluation judge.
# judge_model = OllamaModel(
#     model=os.getenv("OLLAMA_JUDGE_MODEL", "qwen3:8b"),
#     base_url=os.getenv(
#         "OLLAMA_BASE_URL",
#         "http://localhost:11434",
#     ),
#     temperature=0,
# )


class NonThinkingOllamaModel(OllamaModel):
    """DeepEval judge using Ollama with thinking disabled."""

    def generate(self, prompt, schema=None, audio=None):
        chat_model = self.load_model()

        messages = self._prompt_messages(prompt, audio)

        response = chat_model.chat(
            model=self.name,
            messages=messages,
            think=False,
            format=(
                schema.model_json_schema()
                if schema is not None
                else None
            ),
            options={
                "temperature": self.temperature,
                **self.generation_kwargs,
            },
        )

        result = (
            schema.model_validate_json(response.message.content)
            if schema is not None
            else response.message.content
        )

        return result, 0



judge_model = NonThinkingOllamaModel(
    model=os.getenv("OLLAMA_JUDGE_MODEL", "qwen3:8b"),
    base_url=os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ),
    temperature=0,
)


class CustomFaithfulnessTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_verdicts(*args, **kwargs):
        original_prompt = FaithfulnessTemplate.generate_verdicts(
            *args, **kwargs
        )

        return original_prompt + """

        STRICT EVIDENCE-GROUNDING RULES:

        Evaluate every claim using ONLY the supplied
        retrieval context.

        1. A claim is supported only when the evidence
           explicitly states it or clearly entails it.

        2. Absence of contradiction is NOT evidence
           of support.

        3. If a claim introduces a specific algorithm,
           model, number, product, date, or technical fact
           that is not established by the evidence,
           mark the claim as "no".

        4. Do not use pretrained knowledge to fill
           gaps in the retrieval context.

        5. Do not assume a claim is correct simply
           because it sounds plausible.

        EXAMPLE:

        Retrieved evidence:
        "The system retrieves passages from an index."

        Answer claim:
        "The system uses BM25."

        Correct verdict:
        {
            "verdicts": [
                {
                    "verdict": "no",
                    "reason": "The evidence does not identify BM25 as the retrieval algorithm."
                }
            ]
        }

        SEMANTIC ENTAILMENT RULES:

        Evaluate the meaning of claims, not exact wording.

        1. Accept faithful paraphrases and summaries when
           the retrieved evidence clearly entails them.

        2. If the evidence states that a system uses both
           method A and method B, a claim that it combines
           A and B may be supported when "combines" simply
           means both methods are used.

        3. Do NOT infer implementation details that are
           absent from the evidence, such as a specific
           fusion algorithm, weighting scheme, or architecture.

        4. Distinguish between:
           - Supported paraphrases.
           - Unsupported additional technical details.

        EXAMPLE:

        Retrieved evidence:
        "The pipeline uses lexical and dense vector search."

        Answer claim:
        "The pipeline combines lexical and dense retrieval."

        Correct verdict:
        {
            "verdicts": [
                {
                    "verdict": "yes",
                    "reason": "The context establishes that both retrieval methods are used."
                }
            ]
        }

        IMPORTANT JSON OUTPUT REQUIREMENT:

        Every object in the verdicts array MUST include:
        - verdict
        - reason

        Include a reason for ALL verdicts.

        Never omit either field.
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
        verbose_mode=False,
        evaluation_template=CustomFaithfulnessTemplate,
    )

    try:
        metric.measure(test_case)
        print("\n[FAITHFULNESS DEBUG]")
        print("Claims:", getattr(metric, "truths", None))
        print("Assertions:", getattr(metric, "claims", None))
        print("Verdicts:", getattr(metric, "verdicts", None))

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
