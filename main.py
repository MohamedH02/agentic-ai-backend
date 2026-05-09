import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import build_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    task: str
    api_key: str
    model: str = "meta-llama/llama-3.3-70b-instruct:free"
    max_tokens: int = 1024 


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest):
    if not req.api_key.startswith("sk-or"):
        raise HTTPException(status_code=400, detail="api_key must start with 'sk-or'")
    return StreamingResponse(
        event_stream(req.task, req.api_key, req.model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def event_stream(task: str, api_key: str, model: str, max_tokens: int = 1024):
    """Run agent to completion, then replay events cleanly — no partial tokens."""
    try:
        agent = build_agent(api_key, model, max_tokens)

        # Run to full completion (no streaming noise)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=task)]}
        )

        # Replay each message as a clean SSE event
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                continue  # skip the input

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        event = {"type": "action", "tool": tc["name"], "input": tc["args"]}
                        yield f"data: {json.dumps(event)}\n\n"
                        await asyncio.sleep(0.3)   # dramatic reveal
                elif msg.content:
                    is_final = (msg == result["messages"][-1])
                    event = {"type": "final_answer" if is_final else "thought",
                             "content": msg.content}
                    yield f"data: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0.3)

            elif isinstance(msg, ToolMessage):
                event = {"type": "observation", "content": msg.content}
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.3)

        yield 'data: {"type": "done"}\n\n'

    except Exception as e:
        yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
