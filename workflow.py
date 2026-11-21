import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from evidence import evidence_collector
from evaluation import evaluate_faithfulness
from langchain_core.runnables import RunnableConfig


MAX_ATTEMPTS = 3  # Initial attempt + 2 retries


class WorkflowState(TypedDict):
    question: str
    answer: str
    thread_id: str
    evaluation: dict
    attempts: int
    retrieval_context: list[str]


def build_workflow(deep_agent, store):

    def generate_answer(
            state: WorkflowState,
            config: RunnableConfig,
            ):
        attempt = state["attempts"] + 1

        print(f"\n[LANGGRAPH] Generate attempt {attempt}")

        question = state["question"]


        if attempt > 1:
            previous_evaluation = state["evaluation"]
            feedback = previous_evaluation.get("feedback", "")
            checks = previous_evaluation.get("checks", {})

            corrections = []

            if not checks.get("has_evidence", True):
                corrections.append(
                    "MANDATORY: Delegate to research-agent using the "
                    "task tool. The research-agent must call "
                    "rag_search_chunks to retrieve supporting documents."
                )

            if not checks.get("has_citations", True):
                corrections.append(
                    "Include citations in the format [DOC-XXX] "
                    "for claims supported by retrieved documents."
                )

            if not checks.get("valid_citations", True):
                corrections.append(
                    "Use only document IDs returned by the "
                    "retrieval tool. Do not invent citations."
                )

            if previous_evaluation.get("score") is not None:
                if previous_evaluation["score"] < 0.8:
                    corrections.append(
                        "Remove or revise claims unsupported by "
                        "the retrieved evidence."
                    )

            correction_text = "\n".join(corrections)

            question = (
                f"{state['question']}\n\n"
                f"Previous answer:\n{state['answer']}\n\n"
                f"Evaluation feedback:\n{feedback}\n\n"
                f"Required corrections:\n{correction_text}\n\n"
                "Generate a corrected answer."
            )


        # Separate checkpoint history for each attempt
        attempt_thread_id = (
            f"{state['thread_id']}-attempt-{attempt}"
        )

        # Clear evidence from previous executions Otherwise, evidence from previous queries or retry attempts could contaminate the evaluation.
        evidence_collector.clear()

        result = deep_agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config={
                **config,
                "configurable": {
                    **config.get("configurable", {}),
                    "thread_id": attempt_thread_id,
                },
                "recursion_limit": 60,
            },
        )


        # Extract unique documents from retrieval responses
        documents = evidence_collector.get_documents()
        retrieval_context = evidence_collector.get_context()

        print("\n[EVIDENCE DEBUG]")

        for doc in documents:
            print(f"\nDocument ID: {doc['id']}")
            print(f"Title: {doc['title']}")
            print(f"Content: {doc['content'][:200]}")

        print(
            f"\n[EVIDENCE] "
            f"{len(evidence_collector.chunks)} retrieval responses, "
            f"{len(documents)} unique documents"
        )

        return {
            "answer": result["messages"][-1].text,
            "attempts": attempt,
            "retrieval_context": retrieval_context,
        }



    def evaluate_answer(state: WorkflowState):
        print("\n[LANGGRAPH] Evaluate node")

        answer = state["answer"]
        question = state["question"]
        retrieval_context = state["retrieval_context"]

        # Structural checks
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

        cited_ids = {f"[{doc_id}]" for doc_id in citations}
        unknown_citations = sorted(cited_ids - retrieved_ids)

        checks = {
            "non_empty": bool(answer.strip()),
            "has_citations": bool(citations),
            "valid_citations": not unknown_citations,
            "has_evidence": bool(retrieval_context),
        }

        if not all(checks.values()):
            failed_checks = [
                name for name, passed in checks.items()
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
            # DeepEval faithfulness assessment
            faithfulness = evaluate_faithfulness(
                question=question,
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

        return {"evaluation": evaluation}


    def route_after_evaluation(state: WorkflowState):
        if state["evaluation"]["passed"]:
            print("[ROUTER] Passed → END")
            return "end"

        if state["attempts"] >= MAX_ATTEMPTS:
            print("[ROUTER] Retry limit reached → END")
            return "end"

        print("[ROUTER] Failed → Retry generation")
        return "retry"

    builder = StateGraph(WorkflowState)

    builder.add_node("generate", generate_answer)
    builder.add_node("evaluate", evaluate_answer)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", "evaluate")

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