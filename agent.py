import re
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def web_search(query: str) -> str:
    """Search the web for information about a given query."""
    return (
        f"Search results for '{query}': "
        f"According to recent sources, {query} is a rapidly evolving topic with significant developments in 2024. "
        f"Experts highlight key advances including improved efficiency, broader adoption, and new regulatory frameworks. "
        f"Multiple studies confirm that progress in this area continues to accelerate, with major organizations investing heavily."
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


TOOLS = [web_search, calculator]


def build_agent(api_key: str, model: str, max_tokens: int = 1024):
    llm = ChatOpenAI(
        model=model,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.3,
        streaming=True,
        max_tokens=1024,      # ← add this
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState):
        messages = [SystemMessage(content="Think step-by-step, use tools when needed.")] + state["messages"]
        response = llm_with_tools.invoke(messages)
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
