from typing import Annotated, List
from typing_extensions import TypedDict
from operator import add

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END

from llm.open_ai import OpenAIClient


ai_client_manager = OpenAIClient()
llm_client = ai_client_manager.get_llm_client()


@tool(
    "somar",
    description=(
        "Soma dois números. O primeiro número é 'a' "
        "e o segundo é 'b'."
    ),
)
def somar(a: float, b: float) -> str:
    """Soma dois números, 'a' e 'b'."""
    try:
        resultado = a + b
        return str(resultado)
    except Exception as e:
        return f"Erro ao somar: {e}"


tools = [somar]
llm_with_tools = llm_client.bind_tools(tools)

tool_node = ToolNode(tools)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]


def call_llm(state: AgentState):
    response_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [response_msg]}


def call_tool(state: AgentState):
    last_message = state["messages"][-1]
    tool_result_state = tool_node.invoke({"messages": [last_message]})
    return {"messages": tool_result_state["messages"]}


def route_model(state: AgentState):
    last_message = state["messages"][-1]
    return "call_tool" if getattr(last_message, "tool_calls", None) else "end"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm)
workflow.add_node("tool", call_tool)

workflow.add_conditional_edges(
    "agent",
    route_model,
    {"call_tool": "tool", "end": END},
)
workflow.add_edge("tool", "agent")

workflow.set_entry_point("agent")
app = workflow.compile()


if __name__ == "__main__":
    question1 = "Quanto é 15,5 + 42?"
    result1 = app.invoke({"messages": [HumanMessage(content=question1)]})
    print(result1["messages"][-1].content)
    print("-" * 20)

    question2 = "Quanto é a soma de 7.3 e 8.4?"
    result2 = app.invoke({"messages": [HumanMessage(content=question2)]})
    print(result2["messages"][-1].content)
