import asyncio
import os
import json
import sys
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq

load_dotenv()

# Setup Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

async def run_chat():
    # 1. Define how to run your server
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=["server.py"], 
    )

    # 2. Start the server and connect
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 3. Fetch tools
            tools = await session.list_tools()
            print(f"Connected to Server. Tools available: {[t.name for t in tools.tools]}")

            # 4. User Question
            #user_query = "What is 100 + 55? And what is the system status?"
            user_query = input("Ask Query related to Math : ")
            print(f"\nUser Query: {user_query}")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": str(user_query)} 
            ]

            # 5. Send to Groq (First Pass)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in tools.tools]
            )

            # 6. Handle Tool Calls
            tool_calls = response.choices[0].message.tool_calls
            
            if tool_calls:
                # --- THE FIX IS HERE ---
                # We manually build a clean dictionary to avoid sending forbidden fields like 'annotations'
                assistant_msg = response.choices[0].message
                clean_assistant_msg = {
                    "role": "assistant",
                    "content": assistant_msg.content if assistant_msg.content else "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_msg.tool_calls
                    ]
                }
                messages.append(clean_assistant_msg)
                # -----------------------

                for tool_call in tool_calls:
                    print(f"AI Request: Running tool '{tool_call.function.name}'...")
                    
                    # Execute tool on MCP Server
                    args = json.loads(tool_call.function.arguments)
                    result = await session.call_tool(tool_call.function.name, arguments=args)
                    
                    # Feed result back to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.content[0].text)
                    })

                # 7. Get Final Answer (Second Pass)
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages
                )
                print(f"\nAI Answer: {final_response.choices[0].message.content}")
            else:
                print(f"\nAI Answer: {response.choices[0].message.content}")

if __name__ == "__main__":
    asyncio.run(run_chat())
