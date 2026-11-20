
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
    TodoListMiddleware,
    wrap_tool_call,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from knowledge import rag_search_chunks
from workflow import build_workflow
from evidence import evidence_collector


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-3.5-flash-lite"),
    vertexai=False,
    max_retries=1,
    timeout=30,
)


# --------------------------------------------------
# 1. Custom middleware
# --------------------------------------------------



# @wrap_tool_call # It is a Python decorator that registers your function as middleware for tool execution. It enables your function to surround a tool call.
# def log_tool_calls(request, handler):
#     """Log tool name, arguments and execution duration."""

#     import time

#     tool_name = request.tool_call["name"]
#     tool_args = request.tool_call["args"]

#     print(f"\n[TOOL] Name: {tool_name}")
#     print(f"[TOOL] Arguments: {tool_args}")

#     start = time.perf_counter()

#     try:
#         result = handler(request)

#         duration = time.perf_counter() - start # perf_counter() is a high-resolution clock suitable for measuring elapsed time.

#         print(f"[TOOL] Success: {tool_name}")
#         print(f"[TOOL] Duration: {duration:.3f}s")

#         return result # The agent needs the tool result so that Gemini can use it to generate an answer.

#     except Exception as error:
#         print(f"[TOOL] Error: {error}")
#         raise

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

#This line continues the tool execution through the remaining middleware and the tool implementation.

# result = handler(request) This line continues the tool execution through the remaining middleware and the tool implementation.
# Without calling the handler, your code would not execute the tool normally.

# A wrap-style middleware could also:
# - Validate tool arguments before execution.
# - Reject unauthorized operations.
# - Retry specific failures.
# - Modify returned results.
# - Replace execution with a valid alternative response.
# But your current middleware performs logging only.




@wrap_tool_call
def log_tool_calls(request, handler):
    import time

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    print(f"\n[TOOL START] {tool_name}")
    print(f"[ARGS] {tool_args}")

    start = time.perf_counter()

    try:
        result = handler(request)

        duration = time.perf_counter() - start

        # Capture retrieval evidence for evaluation
        if tool_name == "rag_search_chunks":
            content = result.content

            if isinstance(content, str):
                evidence_collector.add(content)

        print(
            f"[TOOL END] {tool_name} "
            f"({duration:.2f}s)"
        )

        return result

    except Exception as error:
        print(f"[TOOL FAILED] {tool_name}: {error}")
        raise


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
# 4. Create specialized subagent
# --------------------------------------------------


research_subagent = {
    "name": "research-agent",

    "description": (
        "Specialist for multi-document research and comparison. "
        "Use for complex requests that require gathering evidence "
        "from multiple knowledge-base documents and synthesizing "
        "a research summary."
    ),

    "system_prompt": """
    You are a specialized research agent.

    Your job:
    1. Search the local knowledge base using rag_search_chunks.
    2. Gather evidence from relevant documents.
    3. Compare findings and identify trade-offs.
    4. Cite document IDs for factual claims.
    5. Return a concise research summary to the supervisor.

    Do not invent source IDs or unsupported findings.
    """,

    "tools": [rag_search_chunks],

    "middleware": [log_tool_calls],
}



# --------------------------------------------------
# 5. Main agent instructions
# --------------------------------------------------


SYSTEM_PROMPT = """
You are a technical Agentic RAG research assistant
acting as the main supervisor agent.

You help users understand retrieval, RAG architecture,
embedding models, reranking, and LLM evaluation.

BEHAVIOR:

1. Simple questions:
   Answer directly if no external evidence is needed.

2. Technical knowledge-base questions:
   You do not have direct access to rag_search_chunks.

   Delegate knowledge-base research to research-agent
   using the task tool.

   Use the evidence returned by research-agent to
   generate your final answer.

3. Complex research questions:
   Delegate document research and comparisons to
   research-agent.

   Provide the subagent with a clear research objective
   and request document IDs as supporting evidence.

   After receiving its findings, synthesize a final answer.

4. Planning:
   For genuinely multi-step tasks, create and update
   a todo list using write_todos.

   Do not create a plan unnecessarily for simple tasks.

5. Filesystem:
   Save temporary research findings under /workspace/.
   Save reusable memory under /memories/.

6. Memory:
   Update /memories/AGENTS.md only when the user
   explicitly asks to remember something reusable.
   Preserve existing instructions.

7. Grounding:
   Cite knowledge-base document IDs supplied by the
   research-agent.

   Never invent source IDs or evaluation scores.
   Admit when evidence is insufficient.

8. Reports:
   For complex research, you may save a report
   using the filesystem write_file tool.

9. Delegation:
   You are responsible for coordinating work,
   not performing knowledge-base retrieval yourself.

   Use research-agent whenever document evidence
   is required.

   You may answer directly when external evidence
   is not necessary.
"""



# --------------------------------------------------
# 6. LangGraph checkpointing
# --------------------------------------------------

checkpointer = MemorySaver()


# --------------------------------------------------
# 7. Create Deep Agent
# --------------------------------------------------

agent = create_deep_agent(
    model=MODEL,

    system_prompt=SYSTEM_PROMPT,

    tools=[], #[rag_search_chunks],

    subagents=[research_subagent],

    middleware=[
        TodoListMiddleware(),
        log_tool_calls,
    ],

    memory=["/memories/AGENTS.md"],

    backend=backend,

    store=store,

    checkpointer=checkpointer,
)

workflow = build_workflow(agent)


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

        config = {
            "configurable": {
                "thread_id": f"local-thread-{thread_number}"
            },
            "recursion_limit": 60, # It helps prevent an agent workflow from running indefinitely, an agent that repeatedly calls tools may eventually exceed the graph recursion limit.
        }

        try:
            result = workflow.invoke({
                "question": question,
                "answer": "",
                "thread_id": f"local-thread-{thread_number}-{uuid4().hex}",
                "evaluation": {},
                "attempts": 0,
                "retrieval_context": [],
            })

            print("\nAttempts:", result["attempts"])
            print("\nEvaluation:", result["evaluation"])

            if result["evaluation"]["passed"]:
                print("\nAgent:")
                print(result["answer"])
            else:
                print("\nAgent:")
                print("Unable to produce a validated answer.")

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


if __name__ == "__main__":
    main()
