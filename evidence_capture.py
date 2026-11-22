# This code is responsible for capturing what your Deep Agent does during execution
# and collecting the documents retrieved by its research subagent.

from threading import Lock
from uuid import UUID
import json

from langchain_core.callbacks import BaseCallbackHandler # LangChain interface that lets your code respond to events during execution.

from evidence import EvidenceCollector


class RequestEvidenceCollector(BaseCallbackHandler):
    """Capture evidence and tool trajectory for one invocation."""

    def __init__(self):
        super().__init__()
        self._lock = Lock() # Protects shared data within this collector from simultaneous modifications by threads
        self._tool_runs: dict[UUID, str] = {} # Maps each running tool's unique ID to its name
        self._outputs: list[str] = [] # Stores raw retrieval responses
        self._trajectory: list[dict] = [] # Stores the sequence and status of tool calls

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs,
    ) -> None:
        # identifies the tool
        tool_name = kwargs.get("name") or (
            serialized.get("name") if serialized else ""
        )

        with self._lock:
            # records the running tool
            self._tool_runs[run_id] = tool_name or ""

            # Record the action, without storing the full
            # prompt or potentially sensitive tool output.
            if tool_name in {
                "task",
                "rag_search_chunks",
                "write_todos",
            }:
                self._trajectory.append({
                    "run_id": str(run_id),
                    "tool": tool_name,
                    "status": "started",
                })

    def on_tool_end(
        self,
        output: object,
        *,
        run_id: UUID,
        **kwargs,
    ) -> None:
        with self._lock:
            tool_name = self._tool_runs.pop(run_id, "")

            for event in self._trajectory:
                if event["run_id"] == str(run_id):
                    event["status"] = "completed"
                    break

            if tool_name != "rag_search_chunks":
                return

            content = getattr(output, "content", output)

            if isinstance(content, str):
                self._outputs.append(content)

            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ):
                        self._outputs.append(block["text"])

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs,
    ) -> None:
        with self._lock:
            self._tool_runs.pop(run_id, None)

            for event in self._trajectory:
                if event["run_id"] == str(run_id):
                    event["status"] = "failed"
                    break

    def get_documents(self) -> list[dict]:
        parser = EvidenceCollector()

        with self._lock:
            outputs = list(self._outputs)

        for output in outputs:
            parser.add(output)

        return parser.get_documents()

    def get_context(self) -> list[str]:
        return [
            f"[{doc['id']}] {doc['title']}\n{doc['content']}"
            for doc in self.get_documents()
        ]

    def get_trajectory(self) -> list[dict]:
        with self._lock:
            return [dict(event) for event in self._trajectory]
