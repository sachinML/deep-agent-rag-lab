import re
from typing import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from evidence_capture import RequestEvidenceCollector
from evaluation import evaluate_faithfulness
from retrieval_evaluation import (
    evaluate_retrieval,
    evaluate_sufficiency,
)
from langchain_core.runnables import RunnableConfig

import time
from litellm.exceptions import RateLimitError


MAX_ATTEMPTS = 2  # Initial attempt + 1 retries

class WorkflowState(TypedDict):
    question: str
    answer: str
    thread_id: str
    evaluation: dict
    attempts: int
    retrieval_context: list[str]
    retrieval_changed: NotRequired[bool]
    trajectory: NotRequired[list[dict]]
    attempt_history: NotRequired[list[dict]]

    # Retrieval evaluation result
    retrieval_evaluation: NotRequired[dict]

    # "retrieval", "generation", or None
    failure_type: NotRequired[str | None]


def run_with_groq_retry(
    evaluation_fn,
    *args,
    max_retries=4,
    **kwargs,
):
    """Retry a rate-limited judge call without regenerating the answer."""

    for attempt in range(max_retries + 1):
        try:
            return evaluation_fn(*args, **kwargs)

        except RateLimitError:
            if attempt == max_retries:
                raise

            delay = min(10 * (2 ** attempt), 60)

            print(
                f"[GROQ RATE LIMIT] Waiting {delay}s "
                f"before retry {attempt + 1}/{max_retries}"
            )

            time.sleep(delay)


def classify_failure(state: WorkflowState) -> str:
    """Classify the next action using retrieval and answer evaluations."""

    retrieval_eval = state.get("retrieval_evaluation", {})
    answer_eval = state.get("evaluation", {})

    sufficient = retrieval_eval.get("sufficient", False)
    answer_passed = answer_eval.get("passed", False)

    if sufficient and answer_passed:
        return "PASSED"

    if sufficient and not answer_passed:
        return "GENERATION_FAILURE"

    if not sufficient:
        return "RETRIEVAL_FAILURE"

    return "UNKNOWN"


def build_failure_retry_message(state: WorkflowState) -> str:
    failure_type = classify_failure(state)

    question = state["question"]
    previous_answer = state.get("answer", "")
    retrieval_context = state.get("retrieval_context", [])
    evaluation = state.get("evaluation", {})
    retrieval_evaluation = state.get("retrieval_evaluation", {})

    if failure_type == "GENERATION_FAILURE":
        context_text = "\n\n".join(retrieval_context)

        return f"""
            The previous answer failed evaluation.

            Original question:
            {question}

            Previous answer:
            {previous_answer}

            Faithfulness feedback:
            {evaluation.get("feedback", "")}

            The retrieved evidence is already sufficient.

            Reuse the following evidence to correct your answer.
            Do not invoke the research agent or retrieve additional documents.

            EXISTING EVIDENCE:
            {context_text}

            Generate a corrected, complete answer using only this evidence.
            Preserve accurate document citations.
            Remove unsupported claims.
            """

    if failure_type == "RETRIEVAL_FAILURE":
                    return f"""
            The previous attempt did not retrieve sufficient evidence.

            Original question:
            {question}

            Missing evidence feedback:
            {retrieval_evaluation.get("sufficiency_feedback", "")}

            Previously retrieved document IDs:
            {retrieval_context}

            Delegate to the research agent to find the missing information.

            Focus on the gaps identified by the sufficiency evaluator.

            Do not invent missing facts.
            If the knowledge base cannot provide the required information,
            explain that limitation.
            """

    return question


def build_workflow(deep_agent, store):

    def generate_answer(
        state: WorkflowState,
        config: RunnableConfig,
    ):
        attempt = state["attempts"] + 1

        print(f"\n[LANGGRAPH] Generate attempt {attempt}")

        previous_context = state.get("retrieval_context", [])
        failure_type = state.get("failure_type")

        if attempt == 1:
            question = state["question"]
            print("[GENERATE] Initial Deep Agent execution")

        else:
            # Different instructions for retrieval and generation failures.
            question = build_failure_retry_message(state)

            print(f"[RETRY] Failure type: {failure_type}")
            print(
                f"[RETRY] Existing evidence: "
                f"{len(previous_context)} passages"
            )

            if failure_type == "GENERATION_FAILURE":
                print("[RETRY] Reuse existing evidence")

            elif failure_type == "RETRIEVAL_FAILURE":
                print("[RETRY] Search for missing evidence")

        attempt_thread_id = (
            f"{state['thread_id']}-attempt-{attempt}"
        )


        # Create a new collector for this invocation only.
        collector = RequestEvidenceCollector()

        # Preserve inherited callbacks, including Langfuse.
        inherited_callbacks = config.get("callbacks")

        if (
            hasattr(inherited_callbacks, "copy")
            and hasattr(inherited_callbacks, "add_handler")
        ):
            # LangGraph may supply a CallbackManager object.
            callbacks = inherited_callbacks.copy()
            callbacks.add_handler(collector, inherit=True)
        else:
            callbacks = [
                *(inherited_callbacks or []),
                collector,
            ]

        result = deep_agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config={
                **config,
                "callbacks": callbacks,
                "configurable": {
                    **config.get("configurable", {}),
                    "thread_id": attempt_thread_id,
                },
                "recursion_limit": 60,
            },
        )

        # Capture retrieval evidence for this invocation.
        new_context = collector.get_context()
        documents = collector.get_documents()

        new_trajectory = collector.get_trajectory()

        trajectory = [
            *state.get("trajectory", []),
            *[
                {**event, "attempt": attempt}
                for event in new_trajectory
            ],
        ]

        print(
            "[TRAJECTORY]",
            [event["tool"] for event in new_trajectory],
        )

        # Preserve old evidence and merge newly retrieved passages.
        retrieval_context = list(previous_context)

        for passage in new_context:
            if passage not in retrieval_context:
                retrieval_context.append(passage)

        print(
            f"[EVIDENCE] "
            f"{len(documents)} newly retrieved documents, "
            f"{len(retrieval_context)} total context passages"
        )

        return {
            "answer": result["messages"][-1].text,
            "attempts": attempt,
            "retrieval_context": retrieval_context,
            "retrieval_changed": retrieval_context != previous_context,
            "trajectory": trajectory,
        }


    def evaluate_retrieved_context(state: WorkflowState):
        print("\n[LANGGRAPH] Evaluate retrieval node")

        # Reuse previous evaluation when the evidence is unchanged.
        if (
            not state.get("retrieval_changed", True)
            and state.get("retrieval_evaluation") is not None
        ):
            print(
                "[RETRIEVAL EVALUATION] "
                "Evidence unchanged → Reusing cached evaluation"
            )

            return {
                "retrieval_evaluation": state["retrieval_evaluation"]
            }

        question = state["question"]
        context = state["retrieval_context"]

        relevancy = run_with_groq_retry(
            evaluate_retrieval,
            question=question,
            retrieval_context=context,
        )

        sufficiency = run_with_groq_retry(
            evaluate_sufficiency,
            question=question,
            retrieval_context=context,
        )

        retrieval_evaluation = {
            "relevant": relevancy["relevant"],
            "relevancy_score": relevancy["score"],
            "relevancy_feedback": relevancy["feedback"],
            "sufficient": sufficiency["sufficient"],
            "sufficiency_score": sufficiency["score"],
            "sufficiency_feedback": sufficiency["feedback"],
        }

        print(
            f"[RETRIEVAL EVALUATION] {retrieval_evaluation}"
        )

        return {
            "retrieval_evaluation": retrieval_evaluation
        }


    def evaluate_answer(state: WorkflowState):
        print("\n[LANGGRAPH] Evaluate node")

        answer = state["answer"]
        retrieval_context = state["retrieval_context"]

        citations = sorted(set(
            re.findall(r"\bDOC-\d{3}\b", answer)
        ))

        retrieved_ids = {
            match
            for passage in retrieval_context
            for match in re.findall(
                r"\[DOC-\d{3}\]", passage
            )
        }

        cited_ids = {
            f"[{doc_id}]" for doc_id in citations
        }

        unknown_citations = sorted(
            cited_ids - retrieved_ids
        )

        checks = {
            "non_empty": bool(answer.strip()),
            "has_citations": bool(citations),
            "valid_citations": not unknown_citations,
            "has_evidence": bool(retrieval_context),
        }

        if not all(checks.values()):
            failed_checks = [
                name
                for name, passed in checks.items()
                if not passed
            ]

            evaluation = {
                "passed": False,
                "score": 0.0,
                "checks": checks,
                "citations": citations,
                "feedback": (
                    f"Structural validation failed: {failed_checks}. "
                    f"Unknown citations: {unknown_citations}"
                ),
            }

        else:
            faithfulness = run_with_groq_retry(
                evaluate_faithfulness,
                question=state["question"],
                answer=answer,
                retrieval_context=retrieval_context,
            )

            evaluation = {
                "passed": faithfulness["passed"],
                "score": faithfulness["score"],
                "checks": checks,
                "citations": citations,
                "feedback": faithfulness["feedback"],
            }

        print(f"[EVALUATION] {evaluation}")

        # Classify using the newly computed evaluation,
        # not the previous attempt's evaluation.
        updated_state = {
            **state,
            "evaluation": evaluation,
        }

        failure_type = classify_failure(updated_state)

        print(f"[FAILURE CLASSIFICATION] {failure_type}")

        # Store the answer and evaluation for this attempt.
        attempt_record = {
            "attempt": state["attempts"],
            "answer": answer,
            "score": evaluation["score"],
            "passed": evaluation["passed"],
            "feedback": evaluation["feedback"],
            "failure_type": failure_type,
            "checks": dict(evaluation["checks"]),
            "citations": list(evaluation["citations"]),
        }

        attempt_history = [
            *state.get("attempt_history", []),
            attempt_record,
        ]

        print(
            f"[ATTEMPT HISTORY] Recorded attempt "
            f"{state['attempts']} "
            f"(score={evaluation['score']:.4f})"
        )

        return {
            "evaluation": evaluation,
            "failure_type": failure_type,
            "attempt_history": attempt_history,
        }

    def print_attempt_history(state: WorkflowState) -> None:
        history = state.get("attempt_history", [])

        print("\n=== GENERATION ATTEMPT HISTORY ===")

        for record in history:
            print(f"\n--- Attempt {record['attempt']} ---")
            print(f"Answer:\n{record['answer']}")
            print(f"Faithfulness score: {record['score']:.4f}")
            print(f"Passed: {record['passed']}")
            print(f"Failure type: {record['failure_type']}")
            print(f"Checks: {record['checks']}")
            print(f"Citations: {record['citations']}")
            print(f"Feedback: {record['feedback']}")

    def route_after_evaluation(state: WorkflowState) -> str:
        failure_type = state.get("failure_type")
        attempts = state.get("attempts", 0)

        if failure_type == "PASSED":
            print("[ROUTER] Passed → END")
            print_attempt_history(state)
            return "end"

        if attempts >= MAX_ATTEMPTS:
            print(
                f"[ROUTER] {failure_type}; "
                "maximum attempts reached → END"
            )
            print_attempt_history(state)
            return "end"

        print(
            f"[ROUTER] {failure_type} → "
            f"Retry attempt {attempts + 1}"
        )

        return "retry"

    # Build the LangGraph workflow.
    builder = StateGraph(WorkflowState)

    builder.add_node("generate", generate_answer)

    builder.add_node(
        "evaluate_retrieval",
        evaluate_retrieved_context,
    )

    builder.add_node("evaluate", evaluate_answer)

    builder.add_edge(START, "generate")

    builder.add_edge(
        "generate",
        "evaluate_retrieval",
    )

    builder.add_edge(
        "evaluate_retrieval",
        "evaluate",
    )

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "retry": "generate",
            "end": END,
        },
    )

    return builder.compile(store=store)



# WorkflowState: Data shared between LangGraph nodes
# StateGraph: LangGraph supports building custom workflows with StateGraph, where nodes are functions that receive state and return updates.
# compile: Creates an executable workflow