# Deep Agent RAG Lab

A hands-on Agentic AI project demonstrating **multi-agent orchestration, Model Context Protocol (MCP), hybrid retrieval-augmented generation (RAG), evidence-grounded answer generation, automated evaluation, deterministic failure recovery, human-in-the-loop approval, and persistent checkpointing**.

Built using **Deep Agents, LangGraph, LangChain, MCP, Elasticsearch, Gemini, Ollama, DeepEval, Langfuse, and SQLite**.

## Overview

This project explores how to build a reliable agentic RAG system that can retrieve evidence, generate cited answers, evaluate its outputs, and recover from retrieval or generation failures.

Rather than relying entirely on LLM reasoning to determine whether and how to retry, the system combines **agentic decision-making with deterministic LangGraph control flow**.

The architecture separates three responsibilities:

- **Deep Agents:** Supervisor orchestration, subagent delegation, and tool selection.
- **MCP:** Standardized discovery and execution of retrieval and service-status tools.
- **LangGraph:** Evaluation, failure classification, bounded recovery, and workflow termination.

### Key Capabilities

- Supervisor agent with a specialized research subagent
- MCP-based tool discovery and execution
- Hybrid Elasticsearch retrieval using BM25 and dense embeddings
- Reciprocal Rank Fusion (RRF) and cross-encoder reranking
- Evidence collection and citation validation
- Retrieval relevancy and sufficiency evaluation
- Answer faithfulness evaluation using DeepEval
- Deterministic retrieval and generation recovery
- Agent tool-trajectory evaluation
- Human-in-the-loop approval for support-ticket creation
- Persistent SQLite checkpointing
- Agent memory using LangGraph Store and Deep Agents backends
- Langfuse observability and tracing

## System Architecture

```text
                        User Question
                              |
                              v
                    LangGraph Workflow
                              |
                              v
                      Generate Answer
                              |
                              v
                    Deep Agent Supervisor
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
           Research Subagent       Supervisor Tools
                  |                       |
                  |                 +-----+------+
                  |                 |            |
                  v                 v            v
           MCP RAG Tool       System Status   Support Ticket
                  |            (via MCP)       (HITL)
                  v
              MCP Client
                  |
                  | JSON-RPC over stdio
                  v
              MCP Server
                  |
                  v
           search_documents()
                  |
                  v
             Elasticsearch
                  |
          +-------+-------+
          |               |
          v               v
       BM25 Search    Dense Search
          |               |
          +-------+-------+
                  |
                  v
           RRF Fusion
                  |
                  v
         Cross-Encoder Reranking
                  |
                  v
         Retrieved Evidence
                  |
                  v
        Research Agent / Supervisor
                  |
                  v
         Candidate Cited Answer
                  |
                  v
          Evidence Collection
                  |
                  v
         Retrieval Evaluation
          |              |
          v              v
       Relevancy      Sufficiency
                  |
                  v
           Answer Evaluation
          |              |
          v              v
       Faithfulness   Citations
                  |
                  v
         Failure Classification
                  |
       +----------+-----------+
       |          |           |
       v          v           v
     PASSED    RETRIEVAL   GENERATION
                FAILURE     FAILURE
       |          |           |
       v          v           v
     Return    Retrieve    Regenerate
     Answer    More Data    With Evidence
                  |           |
                  +-----+-----+
                        |
                        v
                    Reevaluate

```

The workflow uses bounded retry logic to prevent uncontrolled recovery loops.

## Technology Stack

| Component | Technology |
|---|---|
| Agent framework | Deep Agents, LangChain |
| Workflow orchestration | LangGraph |
| Tool interoperability | Model Context Protocol (MCP) |
| MCP transport | stdio |
| Generation model | Google Gemini |
| Local evaluation model | Qwen3 8B via Ollama |
| Search engine | Elasticsearch |
| Sparse retrieval | BM25 |
| Dense retrieval | BGE Sentence Transformers |
| Retrieval fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Document processing | Docling |
| Evaluation | DeepEval |
| Observability | Langfuse |
| Persistent checkpointing | LangGraph SQLite checkpointer |
| Agent memory | LangGraph InMemoryStore |
| Human approval | HumanInTheLoopMiddleware |
| Package management | uv |
| Local infrastructure | Docker Compose |

## Repository Structure

```text
deep-agent-rag-lab/
├── src/
│   └── deep_agent_rag/
│       ├── agents/
│       │   ├── prompts.py
│       │   └── ticket_tools.py
│       ├── mcp_server/
│       │   └── server.py
│       ├── orchestration/
│       │   ├── workflow.py
│       │   ├── hitl.py
│       │   └── checkpoint.py
│       ├── retrieval/
│       │   ├── knowledge.py
│       │   └── ...
│       ├── ingestion/
│       │   └── ...
│       ├── evaluation/
│       │   ├── faithfulness.py
│       │   ├── retrieval.py
│       │   └── trajectory.py
│       └── observability/
│           ├── evidence.py
│           └── evidence_capture.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   └── checkpoints/
├── scripts/
│   ├── test_mcp_connection.py
│   ├── test_mcp_retrieval.py
│   └── ...
├── tests/
├── AGENTS.md
├── main.py
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
└── README.md
```

## How It Works

### 1. Multi-Agent Orchestration

The system uses a Deep Agent supervisor to coordinate tool usage and delegate technical research tasks.

The supervisor has access to a specialized `research-agent`.

For knowledge-base questions:

1. The user submits a question.
2. LangGraph invokes the supervisor.
3. The supervisor delegates research using the `task` tool.
4. The research subagent invokes `rag_search_chunks`.
5. Retrieved evidence is used to produce a cited answer.
6. The outer LangGraph workflow evaluates the answer.

The research subagent is responsible for knowledge retrieval, while the supervisor handles overall task coordination.

### 2. MCP Integration

The project exposes retrieval functionality through a local MCP server.

The MCP server provides two tools:

| Tool | Purpose |
| --- | --- |
| `get_system_status` | Returns basic RAG service status |
| `rag_search_chunks` | Searches indexed research documents |

The research subagent accesses retrieval through an MCP client instead of importing the retrieval tool directly.

The communication flow is:

```text
Research Subagent
       |
       v
LangChain MCP Tool Adapter
       |
       v
MCP Client
       |
       | JSON-RPC / stdio
       v
MCP Server
       |
       v
search_documents()
       |
       v
Elasticsearch
```

The MCP server executes the existing retrieval pipeline and returns document chunks as serialized JSON.

The current implementation uses **stdio transport**, where the MCP server runs as a local Python subprocess.

MCP standardizes tool discovery and invocation but does not replace the underlying retrieval algorithms.

### 3. Hybrid Retrieval

Documents are processed and indexed in Elasticsearch.

The retrieval pipeline combines:

- **BM25:** Keyword-based lexical retrieval.
- **Dense retrieval:** Semantic retrieval using BGE embeddings.
- **RRF:** Fusion of sparse and dense search rankings.
- **Cross-encoder:** Reranking of candidate passages.

The pipeline retrieves candidate passages, reranks them, and returns the highest-ranked chunks.

Each returned chunk includes its identifier, content, and retrieval scores.

### 4. Evidence Collection

A custom callback-based evidence collector captures tool execution events and retrieved document content.

The collector:

- Tracks tool names and execution status.
- Captures retrieved passages.
- Deduplicates evidence using chunk identifiers.
- Maintains retrieval context for evaluation.
- Records agent tool trajectories.

This allows the evaluation workflow to validate answers against retrieved evidence rather than relying solely on the model's self-reported citations.

### 5. Automated Evaluation

The system uses separate retrieval and answer evaluation stages.

## Retrieval evaluation**

- **Context relevancy:** Are the retrieved documents relevant to the question?
- **Context sufficiency:** Is the evidence sufficient to answer the question?

## Answer evaluation**

- **Faithfulness:** Are generated claims supported by the retrieved evidence?
- **Citation validation:** Do cited chunk identifiers exist in the collected evidence?
- **Structural validation:** Is the answer non-empty and appropriately cited?

DeepEval supports faithfulness evaluation using a locally hosted Qwen3 8B model through Ollama.

### 6. Deterministic Failure Recovery

LangGraph classifies execution outcomes and determines the next action.

| Status | Meaning | Action |
| --- | --- | --- |
| `PASSED` | Evidence and answer pass validation | Return answer |
| `RETRIEVAL_FAILURE` | Evidence is insufficient | Retrieve additional evidence and regenerate |
| `GENERATION_FAILURE` | Evidence is sufficient but answer is invalid | Regenerate using existing evidence |

Recovery is controlled by the workflow rather than depending exclusively on LLM decisions.

The current implementation uses a bounded retry mechanism with a maximum of two generation attempts.

### 7. Human-in-the-Loop Approval

The project includes a support-ticket creation workflow with human approval.

The supervisor can invoke `create_support_ticket`, but execution is intercepted by `HumanInTheLoopMiddleware`.

The user must approve or reject the requested action before execution continues.

```text
User requests ticket
         |
         v
Deep Agent Supervisor
         |
         v
create_support_ticket
         |
         v
Human Approval Required
         |
      +--+--+
      |     |
      v     v
   Approve Reject
      |     |
      v     v
   Execute Skip
      |     |
      +--+--+
         |
         v
   Resume Agent
```

This demonstrates how agents can be prevented from executing sensitive actions without explicit authorization.

The current ticket tool is a demonstration implementation rather than a production ticketing-service integration.

### 8. Persistent Checkpointing

The application uses a SQLite-backed LangGraph checkpointer.

Checkpointing supports:

- Persisting execution state.
- Resuming interrupted workflows.
- Human approval and rejection flows.
- Recovering agent execution after a process restart.

Unlike an in-memory checkpointer, SQLite stores checkpoint data on disk.

The application maintains checkpoint data under `data/checkpoints/`.

### 9. Agent Memory

The project uses a LangGraph `InMemoryStore` and Deep Agents filesystem backends.

The backend configuration combines:

- `StateBackend` for state-scoped files.
- `StoreBackend` for memory files under `/memories/`.

The project initializes agent memory using `AGENTS.md`.

This provides reusable context during agent execution.

The current store is process-local and does not independently persist across application restarts.

### 10. Observability and Trajectory Evaluation

Langfuse captures execution traces and evaluation information.

The system also evaluates agent tool trajectories for:

- Required tool usage.
- Failed tool calls.
- Retrieval-call budgets.
- Tool execution status.

For example, a successful RAG request may produce:

```text
task
  |
  v
rag_search_chunks
  |
  v
Answer generation
  |
  v
Evaluation
  |
  v
PASSED
```

Trajectory evaluation verifies that the expected tools were executed and that retrieval remained within the configured budget.

## Getting Started

### Prerequisites

- Python 3.12+
- uv
- Docker
- Ollama
- Google Gemini API credentials

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Create a `.env` file using `.env.example` as a reference.

Configure Gemini credentials, Elasticsearch settings, and optional Langfuse credentials.

Do not commit `.env` files or API keys.

### 3. Start Elasticsearch

```bash
docker compose up -d
```

Ensure that the Elasticsearch index has been populated using the ingestion pipeline before running retrieval queries.

### 4. Start the Local Evaluation Model

```bash
ollama pull qwen3:8b
ollama serve
```

If Ollama is already running, a second server is not required.

### 5. Verify MCP Connectivity

Run:

```bash
uv run python -m scripts.test_mcp_connection
```

Expected discovered tools:

```text
get_system_status
rag_search_chunks
```

### 6. Test MCP Retrieval

Run:

```bash
uv run python -m scripts.test_mcp_retrieval
```

This test verifies that the MCP server can execute the Elasticsearch hybrid retrieval pipeline and return indexed document chunks.

### 7. Run the Application

```bash
uv run python main.py
```

Example question:

> What is retrieval-augmented generation? Explain using the indexed research documents and cite the sources.

The application should delegate retrieval to the research subagent, retrieve evidence through MCP, generate an answer, and evaluate it.

### 8. Test Human Approval

Run the application and enter:

```text
/ticket Create a support ticket for a failed RAG retrieval request.
```

The application will request explicit approval before the ticket tool executes.

### 9. Run Automated Tests

```bash
uv run python -m pytest -q
```
