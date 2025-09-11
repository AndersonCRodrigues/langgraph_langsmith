from llm.google_ai import GoogleAIClient
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from typing import TypedDict


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

if __name__ == "__main__":
    question = "Quem descobriu a América?"
    result = app.invoke({"input": question})
    print(result["output"])
