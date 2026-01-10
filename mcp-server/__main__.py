"""Entry point for running the MCP server via: python -m mcp_server"""

import asyncio
from .server import server


def main():
    """Run the MCP server in stdio mode."""
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
