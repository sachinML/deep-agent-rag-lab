import asyncio
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient(
        {
            "rag_tools": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    "-m",
                    "deep_agent_rag.mcp_server.server",
                ],
            }
        }
    )

    tools = await client.get_tools()

    print("\n=== DISCOVERED MCP TOOLS ===")

    for tool in tools:
        print(tool.name)

    rag_tool = next(
        tool for tool in tools
        if tool.name == "rag_search_chunks"
    )

    print("\n=== CALLING MCP RAG TOOL ===")

    result = await rag_tool.ainvoke(
        {
            "query": "What is retrieval augmented generation?",
            "k": 3,
        }
    )

    print("\n=== MCP RAG RESULT ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
