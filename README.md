# Deep Agent RAG Lab

A hands-on Agentic AI project demonstrating multi-agent orchestration, retrieval-augmented generation (RAG), tool calling, evidence-grounded answer generation, automated evaluation, and deterministic failure recovery.

Built using **Deep Agents, LangGraph, LangChain, Elasticsearch, Gemini, Ollama, and DeepEval**.

## Overview

This project explores how to build a reliable RAG-based AI agent that can retrieve evidence, generate cited answers, evaluate its own outputs, and recover from retrieval or generation failures.

Rather than relying entirely on an LLM to decide whether and how to retry, the system combines **agentic reasoning with deterministic LangGraph control flow**.

Key capabilities:

- Supervisor agent with a specialized research subagent
- Tool-based retrieval from Elasticsearch
- Hybrid retrieval using BM25 and dense embeddings
- Cross-encoder reranking
- Evidence collection and citation validation
- Context relevancy and sufficiency evaluation
- Answer faithfulness evaluation using DeepEval
- Deterministic retrieval and generation recovery
- Agent tool-trajectory evaluation
- Langfuse observability and tracing
- Multi-turn conversation state using LangGraph checkpointing

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
     | task()
     v
Research Subagent
     |
     | rag_search_chunks()
     v
Elasticsearch
     |
     +-- BM25 Search
     +-- Dense Vector Search
     +-- Reciprocal Rank Fusion
     +-- Cross-Encoder Reranking
     |
     v
Retrieved Evidence
     |
     v
Supervisor Generates Cited Answer
     |
     v
Retrieval Evaluation
     |-- Context Relevancy
     |-- Context Sufficiency
     |
     v
Answer Evaluation
     |-- Faithfulness
     |-- Citation Validation
     |
     v
Failure Classification
     |
     +-- PASSED -------------> Final Answer
     |
     +-- RETRIEVAL_FAILURE --> Deterministic Retrieval Recovery
     |                           |
     |                           v
     |                       Regenerate
     |
     +-- GENERATION_FAILURE -> Regenerate Using Existing Evidence
```

The workflow uses a bounded retry mechanism to prevent uncontrolled agent loops.

## Technology Stack

| Component | Technology |
|---|---|
| Agent framework | Deep Agents, LangChain |
| Workflow orchestration | LangGraph |
| Generation model | Google Gemini |
| Local evaluation model | Qwen3 8B via Ollama |
| Search engine | Elasticsearch |
| Sparse retrieval | BM25 |
| Dense retrieval | Sentence Transformers |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder |
| Document processing | Docling |
| Evaluation | DeepEval |
| Observability | Langfuse |
| Checkpointing | LangGraph MemorySaver |
| Package management | uv |
| Local infrastructure | Docker Compose |

## Repository Structure

```text
deep-agent-rag-lab/
├── src/
│   └── deep_agent_rag/
│       ├── agents/
│       │   └── prompts.py
│       ├── orchestration/
│       │   └── workflow.py
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
│   └── evaluation/
├── scripts/
├── tests/
├── main.py
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
└── README.md
```

## How It Works

### 1. Agent Orchestration

The system uses a Deep Agent supervisor that delegates knowledge-retrieval tasks to a specialized `research-agent`.

The research agent accesses indexed documents through the `rag_search_chunks` tool.

The supervisor uses the retrieved evidence to generate answers with source chunk citations.

### 2. Hybrid Retrieval

Documents are processed and indexed in Elasticsearch.

The retrieval pipeline combines:

- **BM25:** Keyword-based lexical retrieval
- **Dense retrieval:** Semantic similarity using embeddings
- **RRF:** Fusion of sparse and dense rankings
- **Cross-encoder:** Reranking of candidate passages

This combines exact-term matching with semantic retrieval.

### 3. Evidence Collection

The system captures retrieved passages and tool-execution information for each generation attempt.

Evidence is deduplicated using chunk identifiers and retained across recovery attempts.

Generated answers reference supporting evidence using chunk citations.

### 4. Automated Evaluation

Two evaluation stages are used.

**Retrieval evaluation**

- Context relevancy: How relevant is the retrieved evidence to the question?
- Context sufficiency: Does the evidence contain enough information to answer the question?

**Answer evaluation**

- Faithfulness: Are the generated claims supported by the retrieved evidence?
- Citation validation: Do cited chunk identifiers exist in the collected evidence?
- Structural checks: Is the answer non-empty and appropriately cited?

DeepEval uses a locally hosted Qwen3 8B model through Ollama for evaluation.

### 5. Deterministic Failure Recovery

LangGraph classifies execution outcomes into three categories.

| Status | Meaning | Action |
|---|---|---|
| `PASSED` | Sufficient evidence and valid answer | Return answer |
| `RETRIEVAL_FAILURE` | Insufficient evidence | Retrieve additional evidence and regenerate |
| `GENERATION_FAILURE` | Evidence sufficient, answer invalid | Regenerate using existing evidence |

Recovery is controlled by the LangGraph workflow rather than relying exclusively on LLM decisions.

The current workflow supports a maximum of two generation attempts.

### 6. Observability and Agent Evaluation

Langfuse is integrated for tracing agent execution.

The project also captures tool trajectories to evaluate:

- Expected tool usage
- Failed tool calls
- Retrieval-call budgets
- Execution across recovery attempts

## Getting Started

### Prerequisites

- Python 3.12+
- uv
- Docker
- Ollama
- Google Gemini API credentials

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Create a `.env` file using `.env.example` as a reference.

Configure the required model credentials, Elasticsearch connection, and optional Langfuse tracing credentials.

Do not commit `.env` or API keys.

### 3. Start Elasticsearch

```bash
docker compose up -d
```

Ensure that the Elasticsearch index has been populated through the project's ingestion pipeline before running retrieval queries.

### 4. Start the local evaluation model

```bash
ollama pull qwen3:8b
ollama serve
```

If Ollama is already running, you do not need to start another server.

### 5. Run the application

```bash
uv run python main.py
```

Example question:

> According to the indexed RAG paper, what is the difference between parametric and non-parametric memory?

### 6. Run tests

```bash
uv run python -m pytest -q
```
