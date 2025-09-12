from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

# from llm.open_ai import OpenAIClient
from llm.google_ai import GoogleAIClient


ai_client_manager = GoogleAIClient()
llm_client = ai_client_manager.get_llm_client()


class AgentState(BaseModel):
    input: str = Field(default="")
    output: str = Field(default="")
    tipo: str = Field(default="")


def realizar_calculo(state: AgentState) -> AgentState:
    return AgentState(
        input=state.input,
        output="Resposta de cáclulo ficticio 42",
    )


def responder_curiosidades(state: AgentState) -> AgentState:
    response = llm_client.invoke([HumanMessage(content=state.input)])
    return AgentState(
        input=state.input,
        output=response.content,
    )


def responder_erro(state: AgentState) -> AgentState:
    return AgentState(
        input=state.input,
        output="Desculpe, não consegui entender sua pergunta.",
    )


def classificar(state: AgentState) -> AgentState:
    pergunta = state.input.lower()
    palavras_chave = {
        "calculo": ["soma", "quanto é", "+", "calcular"],
        "curiosidade": ["quem é", "o que é", "curiosidade", "fato"],
    }

    tipo = "desconhecido"
    for categoria, palavras in palavras_chave.items():
        if any(p in pergunta for p in palavras):
            tipo = categoria
            break

    return AgentState(
        input=state.input,
        output=state.output,
        tipo=tipo,
    )


workflow = StateGraph(AgentState)
workflow.add_node("classificar", classificar)
workflow.add_node("realizar_calculo", realizar_calculo)
workflow.add_node("responder_curiosidades", responder_curiosidades)
workflow.add_node("responder_erro", responder_erro)

workflow.add_conditional_edges(
    "classificar",
    lambda state: [state.tipo],
    {
        "calculo": "realizar_calculo",
        "curiosidade": "responder_curiosidades",
        "desconhecido": "responder_erro",
    },
)

workflow.set_entry_point("classificar")
workflow.set_finish_point(
    [
        "realizar_calculo",
        "responder_curiosidades",
        "responder_erro",
    ]
)

app = workflow.compile()

if __name__ == "__main__":
    perguntas = [
        "Quanto é 2 + 2?",
        "Quem é Albert Einstein?",
        "O que é a teoria da relatividade?",
        "Qual é a capital da França?",
        "Me diga um comando especial",
    ]

    for pergunta in perguntas:
        estado_final = app.invoke(AgentState(input=pergunta, output=""))
        print(f"Pergunta: {pergunta}")
        print(f'Resposta: {estado_final["output"]}\n')
