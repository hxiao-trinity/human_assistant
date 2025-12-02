import asyncio
import uuid
from typing import AsyncGenerator

import streamlit as st
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import os
from pathlib import Path



# =========================
# Ollama + MCP Client
# =========================
class OllamaMCPClient:
    def __init__(self, llm_model: str, embedding_model:str, vlm_model:str, server_url: str, env_vars:dict[str, str]):
        """
        Initialize the MCP client with Ollama LLM.
        
        Args:
            llm_model: Ollama model name (e.g., "gpt-oss:20b-cloud")
            embedding_model: Ollama embedding model (e.g., "nomic-embed-text")
            vlm_model: OpenRouter vision model (e.g., "openrouter/polaris-alpha")
            server_url: MCP server URL
            env_vars: Environment variables (API keys, directories)
        """
        # new code
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vlm_model = vlm_model

        # Extract configuration
        self.data_dir = env_vars.get("DATA_DIR", "")
        self.output_dir = env_vars.get("OUTPUT_DIR", "")
        self.rag_storage_dir = env_vars.get("RAG_STORAGE_DIR", "")
        self.openrouter_api_key = env_vars.get("OPENROUTER_API_KEY", "")
        self.ollama_api_key = env_vars.get("OLLAMA_API_KEY", "")
        
        # old code
        self.server_url = server_url
        self.messages: list[dict] = []
        self.available_tools: list[dict] = []
        
        self.system_prompt = (
            "You are an AI assistant that can use MCP tools exposed by the server.\n"
            "- Only call tools by their exact names from the provided tool list.\n"
            "- Do not invent tool names.\n"
            "- When you call a tool, provide valid arguments matching its schema.\n"
            "- After any tool result, explain the result clearly to the user.\n"
            "- When you call a tool, provide valid arguments matching its schema.\n"
            "- After any tool result, explain the result clearly to the user.\n"
            "- When you want to run a tool, ALWAYS include its exact name.\n"
            "- Do not leave the function name blank.\n"
            "- Do not invent new tool names.\n"
            "- Please use tool to run any command. Please always give tool call as tool_calls object.\n"
            "- You are P3, an intelligent Knowledge Navigator...\n"
            "- When executing query_knowledge, always show responses with source citations.\n"
        )

    async def initialize_tools(self):
        """Initialize connection and fetch tools (one-time operation)."""
        try:
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    response = await session.list_tools()
                    self.available_tools = []
                    for tool in response.tools:
                        self.available_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        })

            self.messages = [{"role": "system", "content": self.system_prompt}]
            return True, f"Connected! Tools available: {len(self.available_tools)}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def _execute_tool_sync(self, tool_call) -> str:
        """Execute tool synchronously by creating a new event loop."""

        async def _do_execute():
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments or {}
                    result = await session.call_tool(tool_name, tool_args)
                    if result.content and len(result.content) > 0 and getattr(result.content[0], "text", None):
                        return result.content[0].text
                    return "Tool executed but returned no content."

        # Create new event loop for each tool execution
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_do_execute())
        finally:
            loop.close()

    def chat_stream(self, user_text: str, max_tool_turns: int = 4):
        """
        Generator that streams responses (converted from async to sync).
        This avoids the async context manager issue with st.write_stream.
        """
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(max_tool_turns):
            stream = ollama.chat(
                model=self.llm_model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            assistant_text = ""
            tool_calls = []

            # Stream tokens as they arrive
            for chunk in stream:
                if chunk.message.content:
                    assistant_text += chunk.message.content
                    yield chunk.message.content
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            self.messages.append(
                {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls or []}
            )

            if not tool_calls:
                return

            # Execute tools synchronously
            yield f"\n\n🔧 Executing {len(tool_calls)} tool(s)...\n\n"

            for tc in tool_calls:
                yield f"🔹 Running: {tc.function.name}\n"
                try:
                    raw = self._execute_tool_sync(tc)
                    tool_call_id = str(uuid.uuid4())

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": (
                            f"The tool '{tc.function.name}' has finished executing.\n"
                            f"Raw output:\n{raw}\n\n"
                            "Now explain this result to the user in a clear, human-readable way."
                        ),
                    })
                except Exception as e:
                    yield f"❌ Tool error: {e}\n"

            # Stream follow-up explanation
            stream2 = ollama.chat(
                model=self.llm_model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            followup_text = ""
            for chunk in stream2:
                if chunk.message.content:
                    followup_text += chunk.message.content
                    yield chunk.message.content

            self.messages.append({"role": "assistant", "content": followup_text})

            if not getattr(chunk.message, "tool_calls", None):
                return

        yield "\n\n⚠️ Couldn't complete within allowed tool-call steps."

  # P3:
  
    async def process_user_message(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Process user message through LLM with tool support.
        Streams response tokens as they arrive.
        
        Args:
            user_input: User's natural language input
        
        Yields:
            Response tokens as strings
        """
        # Your implementation:
        # 1. Add user message to conversation history
        # 2. Call Ollama API with tools enabled
        # 3. Handle tool calls in a loop (agentic behavior)
        # 4. For each response token, yield it
        # 5. Update conversation history

        # 1. Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Up to 5 tool-call cycles
        for _ in range(5):

            # 2. Call LLM with tools enabled
            stream = ollama.chat(
                model=self.llm_model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            assistant_text = ""
            tool_calls = []

            # --- 3. Stream tokens ---
            for chunk in stream:
                m = chunk.get("message", {})
                if "content" in m and m["content"]:
                    assistant_text += m["content"]
                    yield m["content"]

                # Capture tool calls (if any)
                if m.get("tool_calls"):
                    tool_calls.extend(m["tool_calls"])

            # Save assistant message
            self.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls or []
            })

            # --- 4. If no tool calls, we're done ---
            if not tool_calls:
                return

            # --- 5. Execute tools and feed results back ---
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = tc["function"].get("arguments", {})

                # Execute on MCP server
                async with streamablehttp_client(url=self.server_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, tool_args)

                # Extract tool result text
                raw_output = ""
                if result.content and getattr(result.content[0], "text", None):
                    raw_output = result.content[0].text

                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", str(uuid.uuid4())),
                    "content": raw_output,
                })

        # If more than 5 loops → bail out
        yield "\n\n⚠️ Tool-call loop exceeded 5 cycles.\n"
    
    async def call_tool(self, tool_name: str, tool_input: dict):
        """
        Execute a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            tool_input: Arguments for the tool
        
        Returns:
            Tool execution result
        """
        # Your implementation:
        # 1. Connect to MCP server
        # 2. Call session.call_tool()
        # 3. Return result
        response = ollama.chat(
            model=self.llm_model,
            messages=self.messages,
            tools=self.available_tools,
            stream=True  # Enable streaming for responsive UI
        )


    
    def get_conversation_history(self):
        """Return current conversation history."""
        return self.messages
    
    def clear_history(self):
        """Clear conversation history but keep system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]