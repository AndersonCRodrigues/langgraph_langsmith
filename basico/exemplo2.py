from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from llm.open_ai import OpenAIClient


ai_client_manager = OpenAIClient()
llm_client = ai_client_manager.get_llm_client()

system_message = SystemMessage(
    """
    Você é um assistente.
    Se o usuário pedir contas, use a ferramenta 'somar'.
    Caso contrário responda normalmente.
    """.strip()
)


@tool("somar", description="Soma dois números separados por vírgula.")
def somar(valores: str) -> str:
    """Soma dois números separados por vírgula."""
    try:
        a, b = map(float, valores.split(","))
        return str(a + b)
    except Exception as e:
        return f"Erros ao somar: {e}"


tools = [somar]

agent = create_react_agent(
    model=llm_client,
    prompt=system_message,
    tools=tools,
)


def extrair_resposta_final(result):
    ai_message = [
        m for m in result["messages"] if isinstance(m, AIMessage) and m.content
    ]

    return (
        ai_message[-1].content
        if ai_message
        else "Nenhuma resposta encontrada."
    )


if __name__ == "__main__":
    question1 = HumanMessage("Quanto é 15,5 + 42?")
    result = agent.invoke({"messages": [question1]})
    resposta_final = extrair_resposta_final(result)
    print(resposta_final)
    
    question2 = HumanMessage("Quem descobriu a América?")
    result = agent.invoke({"messages": [question2]})
    resposta_final = extrair_resposta_final(result)
    print(resposta_final)
