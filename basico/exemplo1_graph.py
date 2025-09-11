from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph

from llm.google_ai import GoogleAIClient

ai_client_manager = GoogleAIClient()
llm_client = ai_client_manager.get_llm_client()


class State(TypedDict):
    input: str
    output: str


def responder(state: State) -> State:
    input_message = state["input"]
    response = llm_client.invoke(
        input=[HumanMessage(content=input_message)],
    )
    return State(input=input_message, output=response.content)


graph = StateGraph(State)
graph.add_node("responder", responder)
graph.set_entry_point("responder")
graph.set_finish_point("responder")

app = graph.compile()

png_bytes = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

with open("grafo_exemplo1.png", "wb") as f:
    f.write(png_bytes)
