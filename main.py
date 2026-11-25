
import os
from pathlib import Path

from dotenv import load_dotenv
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.backends.utils import create_file_data

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    TodoListMiddleware,
    wrap_tool_call,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deep_agent_rag.retrieval.knowledge import rag_search_chunks
from deep_agent_rag.orchestration.workflow import build_workflow
from deep_agent_rag.evaluation.trajectory import evaluate_trajectory
from deep_agent_rag.agents.ticket_tools import create_support_ticket

from langchain_google_genai import ChatGoogleGenerativeAI

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter

from deep_agent_rag.agents.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    RESEARCH_AGENT_SYSTEM_PROMPT,
)
from langgraph.types import Command
from deep_agent_rag.orchestration.checkpoint import create_sqlite_checkpointer


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

langfuse_handler = CallbackHandler()
langfuse = get_client()


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1, # Model request are spaced about 10 seconds apart.
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

MODEL = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-3.5-flash-lite"),
    vertexai=False,
    max_retries=2,
    timeout=30,
    rate_limiter=rate_limiter,
)


# --------------------------------------------------
# 1. Custom middleware
# --------------------------------------------------

@wrap_tool_call # It is a Python decorator that registers your function as middleware for tool execution. It enables your function to surround a tool call.
def log_tool_calls(request, handler):
    """Log tool name, arguments and execution duration."""

    import time

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    print(f"\n[TOOL START] {tool_name}")
    print(f"[ARGS] {tool_args}")

    start = time.perf_counter()

    try:
        result = handler(request)

        duration = time.perf_counter() - start # perf_counter() is a high-resolution clock suitable for measuring elapsed time.

        print(
            f"[TOOL END] {tool_name} "
            f"({duration:.2f}s)"
        )

        return result # The agent needs the tool result so that Gemini can use it to generate an answer.

    except Exception as error:
        print(f"[TOOL FAILED] {tool_name}: {error}")
        raise

# request contains details about the requested tool call and relevant runtime information.
# handler is the callable that continues the wrapped execution.
# For instance, request.tool_call might contain:
# {
#     "name": "rag_search_chunks",
#     "args": {
#         "query": "RAGAS faithfulness",
#         "k": 3
#     },
#     "id": "call_123",
#     "type": "tool_call"
# }

# This line continues the tool execution through the remaining middleware and the tool implementation.

# result = handler(request) This line continues the tool execution through the remaining middleware and the tool implementation.
# Without calling the handler, your code would not execute the tool normally.

# A wrap-style middleware could also:
# - Validate tool arguments before execution.
# - Reject unauthorized operations.
# - Retry specific failures.
# - Modify returned results.
# - Replace execution with a valid alternative response.
# But your current middleware performs logging only.


# --------------------------------------------------
# 2. Long-term memory store
# --------------------------------------------------

# This creates a LangGraph key-value store in process memory.
# We use it to hold reusable information that can be accessed across different conversation threads.
store = InMemoryStore()

memory_content = (
    BASE_DIR / "AGENTS.md"
).read_text(encoding="utf-8")

store.put(
    ("rag-research-agent",),
    "/AGENTS.md",
    create_file_data(memory_content), # This converts the Markdown string into the file-data representation that the Deep Agents backend expects.
)
# AGENTS.md exists in two different places in your current implementation.
# you copy the original file content into the LangGraph Store.
# If the agent later modifies the stored copy, the local AGENTS.md file on disk does not automatically change.
# This distinction becomes very important when we test long-term memory.

# --------------------------------------------------
# 3. Composite filesystem backend
# --------------------------------------------------

backend = CompositeBackend(
    default=StateBackend(), # When no special route matches the file path, use StateBackend
    routes={
        "/memories/": StoreBackend(   # Any file accessed under /memories/ is routed to StoreBackend
            namespace=lambda runtime: ( # A lambda is a compact way to define a function. is conceptually equivalent to: def get_namespace(runtime): return ("rag-research-agent",)
                "rag-research-agent",
            )
        ),
    },
)
# /workspace/research.md, /workspace/report.md, /notes.md   These paths use StateBackend because none matches /memories/.
# StateBackend stores file content as part of agent state, which can be checkpointed with the thread.
# It does not mean a physical file named research.md appears in your Mac's project directory.

# write_file("/memories/preferences.md", ...) Any file accessed under /memories/ is routed to StoreBackend
# In your current code, the lambda ignores its input and always returns: ("rag-research-agent",). That means all threads using this configuration access the same memory namespace.
# Later, you could implement a user-specific namespace, assuming an authenticated user identity is available in trusted runtime context.


# --------------------------------------------------
# 6. LangGraph checkpointing
# --------------------------------------------------
# persistent memory using sqllite
checkpointer, checkpoint_connection = create_sqlite_checkpointer()

research_subagent = {
    "name": "research-agent",
    "description": (
        "Dedicated knowledge-base retrieval specialist for ALL technical "
        "questions about RAG, retrieval, embeddings, reranking, and "
        "LLM evaluation. Use this subagent for both simple definitions "
        "and complex research tasks. This is the ONLY subagent with "
        "access to rag_search_chunks and Elasticsearch retrieval. "
        "Do not use general-purpose for knowledge-base research."
    ),
    "system_prompt": RESEARCH_AGENT_SYSTEM_PROMPT,
    "tools": [rag_search_chunks],
    "middleware": [log_tool_calls],
}


# --------------------------------------------------
# 7. Create Deep Agent
# --------------------------------------------------

agent = create_deep_agent(
    model=MODEL,

    system_prompt=SUPERVISOR_SYSTEM_PROMPT,

    tools=[create_support_ticket],

    subagents=[research_subagent],

    middleware=[
        TodoListMiddleware(),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "create_support_ticket": {
                    "allowed_decisions": ["approve", "reject"],
                }
            },
            description_prefix="Human approval required",
        ),
        log_tool_calls,
    ],

    memory=["/memories/AGENTS.md"],

    backend=backend,

    store=store,

    checkpointer=checkpointer,
)

workflow = build_workflow(agent, store)


# HITL
def handle_ticket_request(question: str, thread_number: int):
    """Run the HITL-enabled supervisor for a support-ticket request."""

    config = {
        "configurable": {
            "thread_id": f"ticket-{thread_number}-{uuid4().hex}",
        },
        "recursion_limit": 60,
        "callbacks": [langfuse_handler],
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config=config,
    )

    while result.get("__interrupt__"):
        for item in result["__interrupt__"]:
            print("\n=== HUMAN APPROVAL REQUIRED ===")

            for action in item.value.get("action_requests", []):
                print(f"Tool: {action['name']}")
                print(f"Arguments: {action['args']}")

        decision = input(
            "\nApprove ticket creation? (yes/no): "
        ).strip().lower()

        while decision not in ("yes", "no"):
            decision = input(
                "Please enter yes or no: "
            ).strip().lower()

        result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": (
                                "approve"
                                if decision == "yes"
                                else "reject"
                            )
                        }
                    ]
                }
            ),
            config=config,
        )

    print("\nAgent:")

    for message in reversed(result.get("messages", [])):
        if message.type != "ai":
            continue

        content = message.content

        if isinstance(content, str):
            print(content)

        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                ):
                    print(block.get("text", ""))

        break


# --------------------------------------------------
# 8. CLI execution
# --------------------------------------------------

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing."
        )

    print("\nDeep Agent RAG Lab")
    print("Type 'exit' to quit.")
    print("Type '/new' to start a new conversation.\n")
    print("Type '/ticket <description>' to create a support ticket.\n")

    thread_number = 1

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        if question.lower() == "/new":
            thread_number += 1
            print(f"Started conversation {thread_number}")
            continue

        if not question:
            continue

        if question.lower().startswith("/ticket "):
            ticket_description = question[8:].strip()

            if not ticket_description:
                print("Please provide a ticket description.")
                continue

            try:
                handle_ticket_request(
                    question=(
                        "Create a support ticket for the following issue: "
                        + ticket_description
                    ),
                    thread_number=thread_number,
                )
            except Exception:
                import traceback
                traceback.print_exc()

            continue

        if question.lower() == "/ticket":
            print("Usage: /ticket <description>")
            continue

        config = {
            "configurable": {
                "thread_id": f"local-thread-{thread_number}"
            },
            "recursion_limit": 60, # It helps prevent an agent workflow from running indefinitely, an agent that repeatedly calls tools may eventually exceed the graph recursion limit.
        }

        try:
            # generates a practically unique identifier as a 32-character hexadecimal string without hyphens.
            request_id = uuid4().hex

            with langfuse.start_as_current_observation(
                as_type="span", # represents a timed operation, In this case, the operation is your entire RAG request
                name="rag-request",
            ) as span:
                with propagate_attributes(
                    trace_name="deep-agent-rag-workflow",
                    session_id=f"local-thread-{thread_number}",
                    tags=["deepagents", "langgraph", "rag"], # Labels traces for filtering
                ):
                    # print("[DEBUG] workflow type:", type(workflow))
                    # print("[DEBUG] workflow value:", workflow)
                    result = workflow.invoke(
                        {
                            "question": question,
                            "answer": "",
                            "thread_id": (
                                f"local-thread-{thread_number}-{request_id}"
                            ),
                            "evaluation": {},
                            "attempts": 0,
                            "retrieval_context": [],
                        },
                        config={
                            "callbacks": [langfuse_handler], # It receives execution events from LangChain and LangGraph to create observations in your execution trace.
                            "run_name": "agentic-rag-workflow",
                        },
                    )
                    # config carries the settings → callbacks registers observers → CallbackHandler defines what each observer does when events occur.
                    evaluation = result["evaluation"]

                    # Save the quality score on this trace.
                    if evaluation.get("score") is not None:
                        langfuse.score_current_trace(
                            name="faithfulness",
                            value=float(evaluation["score"]),
                            data_type="NUMERIC",
                            comment=evaluation.get("feedback", ""),
                        )

                    # Save whether our evaluation passed.
                    langfuse.score_current_trace(
                        name="evaluation_passed",
                        value=int(bool(evaluation.get("passed", False))),
                        data_type="BOOLEAN",
                    )

                    # Save request/response information.
                    span.update(
                        input={"question": question},
                        output={"answer": result["answer"]},
                        metadata={"attempts": result["attempts"]},
                    )

            langfuse.flush() # send queued data

            print("\nAttempts:", result["attempts"])
            print("\nEvaluation:", result["evaluation"])


            # 1. Display the actual tool execution trajectory
            print("\n=== AGENT TRAJECTORY ===")

            for event in result.get("trajectory", []):
                print(
                    f"Attempt {event['attempt']} | "
                    f"Tool: {event['tool']} | "
                    f"Status: {event['status']}"
                )

            # 2. Evaluate tool usage and efficiency
            trajectory_result = evaluate_trajectory(
                trajectory=result.get("trajectory", []),
                required_tools=["task", "rag_search_chunks"],
                max_retrieval_calls=2,
            )

            print("\n=== TRAJECTORY EVALUATION ===")
            print(trajectory_result)

            # 3. Display the final agent answer
            if result["evaluation"]["passed"]:
                print("\nAgent:")
                print(result["answer"])
            else:
                print("\nAgent:")
                print("Unable to produce a validated answer.")

            # 4. Display the task plan, if one exists
            if result.get("todos"):
                print("\nTask plan:")
                for todo in result["todos"]:
                    print(
                        f"- [{todo['status']}] "
                        f"{todo['content']}"
                    )

        except Exception:
            import traceback
            traceback.print_exc()


        except Exception:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
