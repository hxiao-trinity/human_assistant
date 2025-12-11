import asyncio
import sys
from typing import Optional

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


import ollama
import uuid
class MCPClient:
    def __init__(self, model):
        # Initialize session and client objects
        self.model = model

        # MCP protocol-specific variables: session, exit_stack
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        

    # connecting the mcp client with the server
    async def connect_to_server(self, server_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """

        # 1. set up the connection param for the server                
        # server_params = StdioServerParameters(
        #     command="python3",  # change it to python if your environement supoorts it
        #     args=[server_path],
        #     env=None,
        # )

        ## 2. create the transport layer between the client and the server
        
        # option 1: stdio_client
        # stdio_transport = await self.exit_stack.enter_async_context(
        #   stdio_client(server_params)
        # )
        
        # optoin 2: streamablehttp_client
        transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(url=server_path)
        )

        # 3. create input and output from the transport layer
        self.transport_output, self.transport_input, _ = transport

        #  4. based on the input and output, create a session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.transport_output, self.transport_input)
        )
        print("Successfully connected and initialized session")

        # 5. finally, initialize the session
        await self.session.initialize()

        # 6. query the list of tools avaialable in the server
        response = await self.session.list_tools()

        # 7. iterate over avaiable tools
        self.available_tools = []
        for tool in response.tools:
            # the descriptor of a tool expected by ollama
            tool_descriptor = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            self.available_tools.append(tool_descriptor)

        for tool in self.available_tools:
            print(f"Tool name: {tool['function']['name']}")

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # check whether the client can answer the query
        client_response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=self.available_tools,
            stream=False
        )

        # Process response and handle tool calls
        chat_history = []

        # debug
        # print(response)
        ## model='llama3.2:3b' created_at='2025-09-23T16:21:52.676972Z' done=True done_reason='stop' total_duration=516185084 load_duration=60602209 prompt_eval_count=261 prompt_eval_duration=122292584 eval_count=30 eval_duration=332660375 message=Message(role='assistant', content='', thinking=None, images=None, tool_name=None, tool_calls=[ToolCall(function=Function(name='get_forecast', arguments={'latitude': '40.7128', 'longitude': '-74.0060'}))])

        # update chat history
        if client_response.message.content:
            chat_history.append(client_response.message.content)
            messages.append({
                "role": "assistant",
                "content": client_response.message.content
            })

        # call a tool if needed
        if client_response.message.tool_calls:
            for tool in client_response.message.tool_calls:
                tool_name = tool.function.name
                tool_args = tool.function.arguments

                # Execute tool call on the server
                server_response = await self.session.call_tool(tool_name, tool_args)
                chat_history.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                # debug
                # print(server_response)

                # update chat history
                tool_call_id = str(uuid.uuid4())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content":  server_response.content[0].text
                })

                # parse the servers response into more human readable format
                human_readable_server_response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    # tools=self.available_tools
                )

                if human_readable_server_response.message.content:
                    chat_history.append(human_readable_server_response.message.content)

        # debug
        # print(final_text)
        
        # return the chat history
        return "\n".join(chat_history)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        await self.session.call_tool("initiate_terminal", {"cwd":""})
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python terminal_client.py http://127.0.0.1:3000/mcp")
        sys.exit(1)

    client = MCPClient(model="llama3.2:3b")
    # client = MCPClient(model="gpt-oss:20b")
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())