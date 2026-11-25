def build_retry_message(
    question: str,
    previous_answer: str,
    evaluation: dict,
    retrieval_context: list[str],
) -> str:
    feedback = evaluation.get("feedback", "")
    checks = evaluation.get("checks", {})

    if retrieval_context:
        evidence = "\n\n".join(retrieval_context)
        evidence_status = "Previously retrieved evidence is available."
    else:
        evidence = "No previous evidence available."
        evidence_status = "Research is required before answering."

    return f"""
CORRECTION REQUEST

Original question:
{question}

Previous answer:
{previous_answer}

Evaluation feedback:
{feedback}

Validation checks:
{checks}

Evidence status:
{evidence_status}

Previously retrieved evidence:
{evidence}

Review the evaluation feedback and correct your answer.

Use the available evidence when it is sufficient.
Delegate additional research to research-agent only
if evidence is missing or genuinely insufficient.

Base factual claims on evidence and cite document IDs
in the format [DOC-XXX].

If evidence is insufficient, acknowledge that limitation.
Do not invent citations or unsupported facts.
"""
