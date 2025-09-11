import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


class AnsiColors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.INFO: (
            f"{AnsiColors.GREEN}"
            "%(asctime)s - %(levelname)s - %(message)s"
            f"{AnsiColors.RESET}"
        ),
        logging.ERROR: (
            f"{AnsiColors.RED}"
            "%(asctime)s - %(levelname)s - %(message)s"
            f"{AnsiColors.RESET}"
        ),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(
            record.levelno, "%(asctime)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])


class GoogleAIClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            logging.error(
                "A variavel de ambiente GOOGLE_API_KEY "
                "nao foi definida."
            )
            raise ValueError("GOOGLE_API_KEY nao foi definida.")

    def get_llm_client(self) -> ChatGoogleGenerativeAI:
        logging.info("Inicializando cliente LLM.")
        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model="gemini-2.5-flash-lite",
            temperature=0,
            top_p=0.95,
            max_tokens=2048,
        )

    def get_embedding_client(self) -> GoogleGenerativeAIEmbeddings:
        logging.info("Inicializando cliente de embeddings.")
        return GoogleGenerativeAIEmbeddings(
            google_api_key=self.api_key,
            model="gemini-embedding-001",
        )


if __name__ == "__main__":
    try:
        ai_client_manager = GoogleAIClient()
        llm_client = ai_client_manager.get_llm_client()
        embedding_client = ai_client_manager.get_embedding_client()

        logging.info("Clientes inicializados com sucesso!")

        response = llm_client.invoke("Qual e a capital da França?")
        logging.info(f"Resposta do LLM: {response.content}")

        embeddings = embedding_client.embed_query(
            "Qual e a capital do Brasil?"
        )
        logging.info(
            f"Embedding para 'Qual e a capital do Brasil?': {embeddings[:5]}."
        )

    except ValueError as e:
        logging.error(f"Erro de configuracao: {e}")
    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")
