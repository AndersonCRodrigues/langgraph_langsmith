import os
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from llm.open_ai import OpenAIClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

ai_client_manager = OpenAIClient()
llm_client = ai_client_manager.get_llm_client(model="o4-mini")

system_message = SystemMessage(
    content="""
    Você é um pesquisador muito sarcástico e irônico.
    Use a ferramenta 'search' sempre que necessário, especialmente
    para perguntas que exigem informações atualizadas da web.
    """
)


@tool("search")
def search_web(query: str = "") -> str:
    """
    Busca informações na web baseada na consulta fornecida.

    Args:
        query: A consulta de busca

    Returns:
        As informações encontradas na web ou mensagem de erro
    """
    if not TAVILY_API_KEY:
        return "TAVILY_API_KEY não foi definida."

    if not query.strip():
        return "Consulta de busca não pode estar vazia."

    try:
        tavily_search = TavilySearchResults(max_results=3)
        search_results = tavily_search.invoke(query)
        return search_results or "Nenhuma informação encontrada na web."
    except Exception as e:
        return f"Erro ao buscar na web: {str(e)}"


tools = [search_web]
graph = create_react_agent(
    model=llm_client,
    tools=tools,
    prompt=system_message,
)

export_graph = graph
