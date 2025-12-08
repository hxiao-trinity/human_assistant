# Install dependencies
## For thread-safe audio
```shell
pip3 install playsound
pip3 install gTTS
```

## Install a library to simulate mouse/keyboard events
```shell
pip3 install pyautogui
```

## Install a lightweight OCR:
```shell
pip3 install paddlepaddle
pip3 install paddleocr
```

## Install an Agentic framework based on MCP architecture
https://github.com/lastmile-ai/mcp-agent
```shell
pip3 install mcp-agent
```

## Run the mcp server using:
```shell
python3 p4_mcp_server.py
```

## Run Open-WebUI
```shell
open-webui serve
```
Then go to [localhost:8080](http://localhost:8080) in the Browser



## Add the mcp server to Open-WebUI
### To add an MCP server:
- Open ⚙️ Admin Settings → External Tools.
- Click + (Add Server).
- Set Type to MCP (Streamable HTTP).
- Enter your Server URL: http://127.0.0.1:3000/mcp. Click on the "Verify Connection" button next to it to confirm the connection.
- Auth details to None
- Enter ID to be 1 (or any number) and Name to be equal to the name of the MCP server (e.g., agentic_interaction for P4) 
- Save. If prompted, restart Open WebUI.


## Make change to ollama.py file (only if OpenWebUI throws an error)
Problematic file: /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/open_webui/routers/ollama.py

Replace `send_post_request` function with the following one:

```shell

async def send_post_request(
    url: str,
    payload: Union[str, bytes],
    stream: bool = True,
    key: Optional[str] = None,
    content_type: Optional[str] = None,
    user: UserModel = None,
    metadata: Optional[dict] = None,
):

    r = None
    try:
        session = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        )

        r = await session.post(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {}),
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                        **(
                            {"X-OpenWebUI-Chat-Id": metadata.get("chat_id")}
                            if metadata and metadata.get("chat_id")
                            else {}
                        ),
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        if r.ok is False:
            try:
                # Fix: Handle content type mismatch by reading raw text first
                # then trying to parse as JSON
                text_response = await r.text()
                try:
                    res = json.loads(text_response)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat the text as the error message
                    res = {"error": text_response}

                await cleanup_response(r, session)
                if "error" in res:
                    raise HTTPException(status_code=r.status, detail=res["error"])
            except HTTPException as e:
                raise e  # Re-raise HTTPException to be handled by FastAPI
            except Exception as e:
                log.error(f"Failed to parse error response: {e}")
                raise HTTPException(
                    status_code=r.status,
                    detail=f"Open WebUI: Server Connection Error",
                )

        r.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)
        if stream:
            response_headers = dict(r.headers)

            if content_type:
                response_headers["Content-Type"] = content_type

            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=response_headers,
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            # Fix: Use same approach for non-streaming responses
            text_response = await r.text()
            try:
                res = json.loads(text_response)
            except json.JSONDecodeError:
                # If response isn't JSON, wrap it
                res = {"response": text_response}
            return res

    except HTTPException as e:
        raise e  # Re-raise HTTPException to be handled by FastAPI
    except Exception as e:
        detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status if r else 500,
            detail=detail if e else "Open WebUI: Server Connection Error",
        )
    finally:
        if not stream:
            await cleanup_response(r, session)

```

### Use the following System Prompt for P4 in the OpenWebUI
```shell
You are P4, a UI automation assistant to enable Agentic Interaction.

## Available Tools:

1. **vanilla_workflow_tool(None)** - Simple workflow when you read the .env file
   Use when: User says read .env
   Example: vanilla_workflow_tool()

2. **click_workflow_tool(target:str)** - A workflow when you need to click on a UI element in the screenshot
   Use when: User wants to click on a target (e.g., File, Edit, History)
   Example: click_workflow_tool('History')

3. **capture_screen_with_numbers_tool()** - A workflow when you need to capture a new screenshot
   Use when: User wants to capture a new screenshot
   Example: capture_screen_with_numbers_tool()

4. **echo_tool(command)** - a tool that has a prefix /cc
   Use with /cc: prefix, e.g., /cc: 'start listening'
   Example: echo_tool('start listening')
```

### Optional: VLM model P4
Feel free to use any other VLM models from OpenRouter. For example, we found that 
https://openrouter.ai/nvidia/nemotron-nano-12b-v2-vl:free (name: nvidia/nemotron-nano-12b-v2-vl:free) is quite good.
