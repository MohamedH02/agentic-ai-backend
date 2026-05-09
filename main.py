import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
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


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest):
    if not req.api_key.startswith("sk-or"):
        raise HTTPException(status_code=400, detail="api_key must start with 'sk-or'")
    return StreamingResponse(
        event_stream(req.task, req.api_key, req.model, req.max_tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def event_stream(task: str, api_key: str, model: str, max_tokens: int = 1024):
    try:
        agent = build_agent(api_key, model, max_tokens)
        streaming_text = False
        has_tool_calls = False

        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=task)]},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_start":
                streaming_text = False
                has_tool_calls = False

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # Detect tool-call chunks (means this turn is not a text response)
                if getattr(chunk, "tool_call_chunks", None):
                    has_tool_calls = True
                # Stream text tokens as "thought" (may be reclassified below)
                if chunk.content and not has_tool_calls:
                    streaming_text = True
                    yield sse({"type": "thought", "content": chunk.content})

            elif kind == "on_chat_model_end":
                output = event["data"]["output"]
                # If we streamed text and there are no tool calls, it's the final answer
                if streaming_text and not output.tool_calls:
                    yield sse({"type": "reclassify_last", "to": "final_answer"})
                # Emit each tool call as an action event
                if output.tool_calls:
                    for tc in output.tool_calls:
                        yield sse({"type": "action", "tool": tc["name"], "input": tc["args"]})
                streaming_text = False
                has_tool_calls = False

            elif kind == "on_tool_end":
                raw = event["data"].get("output", "")
                content = raw.content if hasattr(raw, "content") else str(raw)
                yield sse({"type": "observation", "content": content})

        yield 'data: {"type":"done"}\n\n'

    except Exception as e:
        yield sse({"type": "error", "message": str(e)})
