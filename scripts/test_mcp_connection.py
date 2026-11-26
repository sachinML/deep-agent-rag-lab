# connects to the server, discovers available tools, and executes one
import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient # MCP client provided by LangChain's MCP adapters

# The client and server communicate through the subprocess's stdin and stdout streams.
# You don't need to start the server manually. The MCP client launches it when establishing the stdio connection.

async def main():
    client = MultiServerMCPClient(
        {
            "rag_tools": { # Logical name of this MCP server in the client configuration
                "transport": "stdio", # Communicate through standard input and output
                "command": sys.executable, # Use the Python executable running the client
                "args": [ # Arguments passed to that Python executable
                    "-m",         # Run a Python module
                    "deep_agent_rag.mcp_server.server", # Module containing our MCP server
                ],
            }
        }
    )
    # Suppose your current Python executable is: /Users/sachinjha/Code/deep-agent-rag-lab/.venv/bin/python
    # Then the client effectively starts a subprocess equivalent to: python -m deep_agent_rag.mcp_server.server

    # await means: suspend this coroutine until the operation completes, allowing the event loop to run other work in the meantime
    tools = await client.get_tools()

    print("\n=== DISCOVERED MCP TOOLS ===")

    for tool in tools:
        print(f"Name: {tool.name}")
        print(f"Description: {tool.description}")

    status_tool = next( # next() raises StopIteration if no matching tool exists.
        tool for tool in tools
        if tool.name == "get_system_status"
    )

    result = await status_tool.ainvoke({}) # langChain's asynchronous tool invocation method, {} represents the tool's input arguments.

    print("\n=== MCP TOOL RESULT ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
