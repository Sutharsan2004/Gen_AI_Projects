from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-mcp-server")

@mcp.tool()
def calc(a: int, b:int) -> int:
    return a+b
@mcp.tool()
def status_checker() -> str:
    return "Server is running 100% successfully.."

if __name__ == '__main__':
    mcp.run()
