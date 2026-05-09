import json
import httpx
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current, real-world information about a topic."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return f"No results found for: {query}"
        return "\n\n".join(
            f"Title: {r['title']}\nSummary: {r['body']}\nURL: {r['href']}"
            for r in results
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression using +, -, *, /, (, )."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


TOOLS = [web_search, calculator]

SYS = (
    "You are a helpful research assistant. Think step-by-step. "
    "Use web_search to find current, real-world information and always cite URLs. "
    "Use calculator for any arithmetic. Be concise but thorough."
)


# ── HTTP transport that writes max_tokens directly into the raw JSON body ──────
# The openai SDK 1.x converts max_tokens→max_completion_tokens when tools are
# present. Some OpenRouter providers (e.g. Venice) ignore max_completion_tokens
# and fall back to context_window-prompt_tokens (~58k), which exceeds their
# output limit. Intercepting at the transport level guarantees the field lands
# in the actual bytes sent over the wire, no matter what the SDK does above.

class _CapMaxTokens(httpx.AsyncBaseTransport):
    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.content:
            try:
                body = json.loads(req.content)
                body["max_tokens"] = self._cap
                payload = json.dumps(body).encode()
                # Rebuild request; httpx auto-sets content-length from content=
                headers = [
                    (k, v) for k, v in req.headers.multi_items()
                    if k.lower() != "content-length"
                ]
                req = httpx.Request(req.method, req.url,
                                    headers=headers, content=payload)
            except Exception:
                pass
        return await self._inner.handle_async_request(req)


# ── Agent builder ───────────────────────────────────────────────────────────────

def build_agent(api_key: str, model: str, max_tokens: int = 1024):
    safe_max = min(max_tokens, 8192)

    async_http = httpx.AsyncClient(transport=_CapMaxTokens(safe_max))

    llm = ChatOpenAI(
        model=model,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.3,
        streaming=True,
        http_async_client=async_http,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    async def agent_node(state: AgentState):
        messages = [SystemMessage(content=SYS)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def route(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
