
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate
from deepeval.test_case import LLMTestCase

from evaluation import judge_model
from pydantic import BaseModel, Field


RETRIEVAL_RELEVANCY_THRESHOLD = 0.7


class CustomContextualRelevancyTemplate(ContextualRelevancyTemplate):
    @staticmethod
    def generate_verdicts(input: str, context: str):
        return f"""
Evaluate whether each factual statement in the
retrieved context is relevant to answering the
user's question.

Return ONLY a valid JSON object.

The JSON object must have a "verdicts" array.

EVERY item in the array MUST contain ALL THREE fields:

1. "statement": the statement being evaluated.
2. "verdict": either "yes" or "no".
3. "reason": a non-empty explanation of the verdict.

IMPORTANT:
The "reason" field is MANDATORY for BOTH
"yes" and "no" verdicts.

Never omit "reason", even when the statement
is clearly relevant.

Example:

{{
    "verdicts": [
        {{
            "statement": "BM25 uses lexical matching.",
            "verdict": "yes",
            "reason": "This describes the retrieval method asked about."
        }},
        {{
            "statement": "Kubernetes manages containers.",
            "verdict": "no",
            "reason": "This is unrelated to the retrieval question."
        }}
    ]
}}

USER QUESTION:
{input}

RETRIEVED CONTEXT:
{context}

JSON:
"""


def evaluate_retrieval(
    question: str,
    retrieval_context: list[str],
) -> dict:

    if not retrieval_context:
        return {
            "relevant": False,
            "score": 0.0,
            "feedback": "No retrieval evidence was available.",
        }

    test_case = LLMTestCase(
        input=question,
        actual_output="N/A",
        retrieval_context=retrieval_context,
    )

    metric = ContextualRelevancyMetric(
        threshold=RETRIEVAL_RELEVANCY_THRESHOLD,
        model=judge_model,
        include_reason=True,
        async_mode=False,
        verbose_mode=False,
        evaluation_template=CustomContextualRelevancyTemplate,
    )

    metric.measure(test_case)

    return {
        "relevant": bool(metric.is_successful()),
        "score": metric.score,
        "feedback": metric.reason or "",
    }




class EvidenceSufficiencyVerdict(BaseModel):
    sufficient: bool = Field(
        description="Whether the evidence is sufficient."
    )
    reason: str = Field(
        description="Explanation of the sufficiency decision."
    )


def evaluate_sufficiency(
    question: str,
    retrieval_context: list[str],
) -> dict:
    """Evaluate whether retrieved evidence can answer the question."""

    if not retrieval_context:
        return {
            "sufficient": False,
            "score": 0.0,
            "feedback": "No retrieved evidence is available.",
        }

    context_text = "\n\n".join(retrieval_context)

    prompt = f"""
        You are an evidence sufficiency evaluator for a RAG system.

        Your task is to determine whether the retrieved evidence
        contains enough information to answer the user's question.

        Evaluation rules:

        1. Identify all essential parts of the user's question.

        2. Evaluate ONLY the supplied retrieved evidence.
        Do not use your own knowledge to fill missing information.

        3. Return sufficient=true ONLY if the evidence contains
        enough facts to answer all essential parts of the question.

        4. Return sufficient=false if important information is
        missing, even when some retrieved passages are relevant.

        5. For comparison questions, both subjects must have
        adequate supporting evidence.

        6. An answer saying "I don't know" does not make
        the retrieved evidence sufficient.

        7. Explain your decision clearly.

        Return a JSON object with BOTH required fields:
        - "sufficient": boolean
        - "reason": non-empty explanation

        Do not omit either field.

        USER QUESTION:
        {question}

        RETRIEVED EVIDENCE:
        {context_text}
        """


    result = judge_model.generate(
        prompt,
        schema=EvidenceSufficiencyVerdict,
    )

    # Accommodate DeepEval model adapters that return
    # either the verdict or (verdict, cost).
    verdict = result[0] if isinstance(result, tuple) else result

    if isinstance(verdict, dict):
        verdict = EvidenceSufficiencyVerdict.model_validate(verdict)

    if not isinstance(verdict, EvidenceSufficiencyVerdict):
        raise TypeError(
            "Unexpected sufficiency judge output: "
            f"{type(verdict).__name__}"
        )

    if not verdict.reason.strip():
        raise ValueError(
            "Sufficiency judge returned an empty reason."
        )

    return {
        "sufficient": verdict.sufficient,
        "score": 1.0 if verdict.sufficient else 0.0,
        "feedback": verdict.reason,
    }

