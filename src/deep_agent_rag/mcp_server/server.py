from mcp.server.fastmcp import FastMCP

from deep_agent_rag.retrieval.knowledge import search_documents
import json


mcp = FastMCP("deep-agent-rag-tools")


@mcp.tool()
def get_system_status() -> dict:
    """Return the status of the RAG service."""
    return {
        "service": "deep-agent-rag",
        "status": "healthy",
        "version": "1.0",
    }


@mcp.tool()
def rag_search_chunks(query: str, k: int = 3) -> str:
    """
    Search indexed research documents using hybrid
    BM25 + dense retrieval and cross-encoder reranking.

    Returns a JSON array of chunks with citation IDs.
    """
    documents = search_documents(query=query, k=k)

    return json.dumps(documents, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")

# mcp.run(transport="stdio") #starts the server using standard input/output communication
# Unlike an HTTP server, stdio doesn't require opening a network port.
# The MCP client launches the server process and communicates with it through stdin and stdout.
