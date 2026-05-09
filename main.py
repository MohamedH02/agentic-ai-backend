import json
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


async def event_stream(task: str, api_key: str, model: str):
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    try:
        agent = build_agent(api_key, model, max_tokens=1024)
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=task)]},
            stream_mode="messages",
        ):
            msg, metadata = chunk
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield sse({"type": "action", "tool": tc["name"], "input": tc["args"]})
                elif msg.content:
                    ev_type = "final_answer" if not msg.tool_calls else "thought"
                    yield sse({"type": ev_type, "content": msg.content})
            elif isinstance(msg, ToolMessage):
                yield sse({"type": "observation", "content": msg.content})
    except Exception as e:
        yield sse({"type": "error", "message": str(e)})
    finally:
        yield 'data: {"type":"done"}\n\n'
