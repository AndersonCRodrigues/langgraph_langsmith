import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


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


class OpenAIClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logging.error(
                "A variavel de ambiente OPENAI_API_KEY nao foi definida."
            )
            raise ValueError("OPENAI_API_KEY nao foi definida.")

    def get_llm_client(self) -> ChatOpenAI:
        logging.info("Inicializando cliente LLM.")
        return ChatOpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo-0125",
            temperature=0,
            max_tokens=2048,
        )


if __name__ == "__main__":
    try:
        ai_client_manager = OpenAIClient()
        llm_client = ai_client_manager.get_llm_client()

        logging.info("Cliente LLM inicializado com sucesso!")

        response = llm_client.invoke("Qual e a capital da França?")
        logging.info(f"Resposta do LLM: {response.content}")

    except ValueError as e:
        logging.error(f"Erro de configuracao: {e}")
    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")
