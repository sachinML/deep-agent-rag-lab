import re
import json
from deep_agent_rag.retrieval.knowledge import rag_search_chunks
from typing import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from deep_agent_rag.observability.evidence_capture import RequestEvidenceCollector
from deep_agent_rag.evaluation.faithfulness import evaluate_faithfulness
from deep_agent_rag.evaluation.retrieval import (
    evaluate_retrieval,
    evaluate_sufficiency,
)
from langchain_core.runnables import RunnableConfig


MAX_ATTEMPTS = 2  # Initial attempt + 1 retries

class WorkflowState(TypedDict):
    question: str
    answer: str
    thread_id: str
    response_type: NotRequired[str]
    tool_result: NotRequired[dict]
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
    recovery_changed: NotRequired[bool]




def classify_failure(state: WorkflowState) -> str:
    """Classify failures based on evidence sufficiency and answer quality."""

    retrieval_eval = state.get("retrieval_evaluation", {})
    answer_eval = state.get("evaluation", {})

    sufficient = retrieval_eval.get("sufficient", False)
    answer_passed = answer_eval.get("passed", False)

    # Insufficient evidence means another retrieval may be necessary.
    if not sufficient:
        return "RETRIEVAL_FAILURE"

    # Sufficient evidence, but answer failed validation.
    if not answer_passed:
        return "GENERATION_FAILURE"

    # Enough evidence and a validated answer.
    return "PASSED"



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
        context_text = "\n\n".join(retrieval_context)

        return f"""
        The previous attempt had insufficient evidence.

        Original question:
        {question}

        Missing evidence feedback:
        {retrieval_evaluation.get("sufficiency_feedback", "")}

        LangGraph has already performed an additional
        retrieval operation.

        Use the complete accumulated evidence below.
        Do not invoke the research agent again.

        RETRIEVED EVIDENCE:
        {context_text}

        Generate a corrected answer using only this evidence.
        Cite the exact chunk IDs.

        If the evidence is still insufficient, explicitly
        acknowledge the missing information.
        """


    return question


def build_workflow(deep_agent, store):

    def recover_retrieval(state: WorkflowState):
        """Force retrieval after an evidence-sufficiency failure."""

        print("\n[LANGGRAPH] Deterministic retrieval recovery")

        question = state["question"]
        feedback = state.get(
            "retrieval_evaluation", {}
        ).get("sufficiency_feedback", "")

        query = question

        if feedback:
            query += f"\nMissing information: {feedback[:250]}"

        # Explicit retrieval: no LLM tool-selection decision.
        raw_result = rag_search_chunks.invoke(
            {"query": query, "k": 8}
        )

        documents = (
            json.loads(raw_result)
            if isinstance(raw_result, str)
            else raw_result
        )

        if not isinstance(documents, list):
            raise ValueError(
                "Retrieval recovery expected a list of documents"
            )

        existing_context = list(
            state.get("retrieval_context", [])
        )

        existing_ids = {
            passage.split("]", 1)[0].removeprefix("[")
            for passage in existing_context
            if passage.startswith("[") and "]" in passage
        }

        new_count = 0

        for document in documents:
            chunk_id = document.get(
                "chunk_id", document.get("id")
            )

            if not chunk_id:
                continue

            if chunk_id in existing_ids:
                continue

            title = document.get("title", "")
            content = document.get("content", "")

            passage = f"[{chunk_id}] {title}\n{content}"

            existing_context.append(passage)
            existing_ids.add(chunk_id)
            new_count += 1

        attempt = state["attempts"] + 1

        trajectory = [
            *state.get("trajectory", []),
            {
                "tool": "rag_search_chunks",
                "status": "completed",
                "attempt": attempt,
                "source": "workflow_recovery",
            },
        ]

        print(
            f"[RECOVERY] {new_count} new chunks; "
            f"{len(existing_context)} total passages"
        )

        return {
            "retrieval_context": existing_context,
            "recovery_changed": new_count > 0,
            "trajectory": trajectory,
        }


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
                print("[RETRY] Regenerate using recovered evidence")

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

        # This checks the actual tool response rather than trusting the model's claim that it called the tool.
        status_tool_results = [
            message
            for message in result["messages"]
            if getattr(message, "type", None) == "tool"
            and getattr(message, "name", None) == "get_system_status"
        ]

        response_type = "rag"
        tool_result = {}

        if status_tool_results:
            latest_result = status_tool_results[-1]

            try:
                content = latest_result.content

                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)

                if not isinstance(content, str):
                    raise ValueError("Unexpected MCP response format")

                parsed_result = json.loads(content)

                if (
                    isinstance(parsed_result, dict)
                    and parsed_result.get("service") == "deep-agent-rag"
                    and parsed_result.get("status")
                    in {"healthy", "degraded", "unhealthy"}
                    and isinstance(parsed_result.get("version"), str)
                ):
                    response_type = "tool_status"
                    tool_result = parsed_result

            except (ValueError, TypeError, json.JSONDecodeError):
                print("[MCP] Status response could not be validated")

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
            "retrieval_changed": (
                retrieval_context != previous_context
                or state.get("recovery_changed", False)
            ),
            "recovery_changed": False,
            "trajectory": trajectory,
            "response_type": response_type,
            "tool_result": tool_result,
        }

    # This validates the MCP result without requiring document citations.
    def evaluate_tool_response(state: WorkflowState):
        print("\n[LANGGRAPH] Evaluate MCP tool response")

        tool_result = state.get("tool_result", {})

        passed = (
            tool_result.get("service") == "deep-agent-rag"
            and tool_result.get("status")
            in {"healthy", "degraded", "unhealthy"}
            and isinstance(tool_result.get("version"), str)
        )

        evaluation = {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "checks": {
                "valid_tool_response": passed,
            },
            "citations": [],
            "feedback": (
                "MCP status tool returned a valid response."
                if passed
                else "MCP status response validation failed."
            ),
        }

        attempt_record = {
            "attempt": state["attempts"],
            "answer": state["answer"],
            "score": evaluation["score"],
            "passed": evaluation["passed"],
            "feedback": evaluation["feedback"],
            "failure_type": (
                "PASSED" if passed else "TOOL_FAILURE"
            ),
            "checks": evaluation["checks"],
            "citations": [],
        }

        print(f"[TOOL EVALUATION] {evaluation}")

        return {
            "evaluation": evaluation,
            "failure_type": (
                "PASSED" if passed else "TOOL_FAILURE"
            ),
            "attempt_history": [
                *state.get("attempt_history", []),
                attempt_record,
            ],
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

        relevancy = evaluate_retrieval(
            question=question,
            retrieval_context=context,
        )

        sufficiency = evaluate_sufficiency(
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

        # Support synthetic DOC IDs and Elasticsearch chunk IDs.
        citation_pattern = r"(?:DOC-\d{3}|[a-f0-9]{16}-\d{4})"

        # Extract individual IDs, including grouped citations:
        # [ID1, ID2] -> [ID1, ID2]
        citation_groups = re.findall(
            r"\[([^\[\]]+)\]",
            answer,
        )

        citations = set()
        malformed_citations = []

        for group in citation_groups:
            # Only process brackets that appear to contain
            # our document/chunk citation format.
            if not re.search(r"DOC-|[a-f0-9]{16}-", group):
                continue

            for item in group.split(","):
                item = item.strip()

                if re.fullmatch(citation_pattern, item):
                    citations.add(item)
                else:
                    malformed_citations.append(item)

        citations = sorted(citations)

        retrieved_ids = {
            match
            for passage in retrieval_context
            for match in re.findall(
                rf"\[({citation_pattern})\]",
                passage,
            )
        }

        unknown_citations = sorted(
            set(citations) - retrieved_ids
        )

        checks = {
            "non_empty": bool(answer.strip()),
            "has_citations": bool(citations),
            "valid_citations": (not unknown_citations and not malformed_citations),
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
                    f"Unknown citations: {unknown_citations}. "
                    f"Malformed citations: {malformed_citations}"
                ),
            }

        else:
            faithfulness = evaluate_faithfulness(
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

        if failure_type == "RETRIEVAL_FAILURE":
            print(
                "[ROUTER] Retrieval failure → "
                "deterministic retrieval recovery"
            )
            return "recover"

        print(
            "[ROUTER] Generation failure → "
            f"retry attempt {attempts + 1}"
        )
        return "retry"

    def route_after_generation(state: WorkflowState) -> str:
        if state.get("response_type") == "tool_status":
            print("[ROUTER] MCP status response → Tool evaluation")
            return "tool"

        print("[ROUTER] Research response → RAG evaluation")
        return "rag"


    # Build the LangGraph workflow.
    builder = StateGraph(WorkflowState)

    builder.add_node("generate", generate_answer)
    builder.add_node("recover_retrieval", recover_retrieval)

    builder.add_node(
        "evaluate_retrieval",
        evaluate_retrieved_context,
    )

    builder.add_node("evaluate", evaluate_answer)
    builder.add_node("evaluate_tool", evaluate_tool_response)

    builder.add_edge(START, "generate")

    builder.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "tool": "evaluate_tool",
            "rag": "evaluate_retrieval",
        },
    )

    builder.add_edge("evaluate_tool", END)

    builder.add_edge(
        "evaluate_retrieval",
        "evaluate",
    )

    builder.add_edge("recover_retrieval", "generate")

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "retry": "generate",
            "recover": "recover_retrieval",
            "end": END,
        },
    )

    return builder.compile(store=store)



# WorkflowState: Data shared between LangGraph nodes
# StateGraph: LangGraph supports building custom workflows with StateGraph, where nodes are functions that receive state and return updates.
# compile: Creates an executable workflow