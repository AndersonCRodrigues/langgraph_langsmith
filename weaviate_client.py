import weaviate
import logging
from weaviate.connect import ConnectionParams, ProtocolParams
from typing import Optional


class AnsiColors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.INFO: f"{AnsiColors.GREEN}%(asctime)s - %(levelname)s - %(message)s{AnsiColors.RESET}",
        logging.ERROR: f"{AnsiColors.RED}%(asctime)s - %(levelname)s - %(message)s{AnsiColors.RESET}",
        logging.WARNING: f"{AnsiColors.YELLOW}%(asctime)s - %(levelname)s - %(message)s{AnsiColors.RESET}",
        logging.DEBUG: "%(asctime)s - %(levelname)s - %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])


class WeaviateConnection:
    def __init__(
        self,
        http_host: str = "localhost",
        http_port: int = 8080,
        grpc_host: str = "localhost",
        grpc_port: int = 50051,
        secure: bool = False,
    ):
        http_params = ProtocolParams(
            host=http_host,
            port=http_port,
            secure=secure,
        )
        grpc_params = ProtocolParams(
            host=grpc_host,
            port=grpc_port,
            secure=secure,
        )
        self.connection_params = ConnectionParams(
            http=http_params,
            grpc=grpc_params,
        )
        self.client: Optional[weaviate.WeaviateClient] = None

    def connect(self):
        if self.is_connected():
            logging.info("Ja esta conectado.")
            return

        logging.info("Iniciando a conexao com Weaviate...")
        try:
            self.client = weaviate.WeaviateClient(
                connection_params=self.connection_params
            )
            self.client.connect()
            if self.is_connected():
                logging.info("Conectado com sucesso!")
            else:
                raise ConnectionError("Falha ao estabelecer conexao com Weaviate.")
        except ConnectionError as e:
            logging.error(f"Erro de conexao: {e}")
            logging.info("Verifique se o Weaviate esta rodando.")
            self.disconnect()
            raise
        except Exception as e:
            logging.error(f"Erro inesperado na conexao: {e}")
            self.disconnect()
            raise

    def disconnect(self):
        if self.client:
            logging.info("Desconectando de Weaviate...")
            self.client.close()
            self.client = None
            logging.info("Desconectado.")
        else:
            logging.info("Cliente nao esta conectado.")

    def is_connected(self) -> bool:
        return self.client and self.client.is_connected()

    def get_client(self) -> Optional[weaviate.WeaviateClient]:
        if self.is_connected():
            return self.client
        return None

    def __enter__(self):
        self.connect()
        return self.get_client()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


if __name__ == "__main__":
    logging.info("=== Exemplo de uso com metodos connect() e disconnect() ===")
    conn = WeaviateConnection()
    try:
        conn.connect()
        if conn.is_connected():
            client = conn.get_client()
            if client:
                info = client.get_meta()
                logging.info(f"Versao do Weaviate: {info.get('version', 'N/A')}")
        conn.disconnect()
    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")

    logging.info("\n---")
    logging.info("=== Exemplo de uso com 'with' ===")
    try:
        with WeaviateConnection() as client:
            logging.info("Conexao estabelecida, o cliente esta disponivel.")
            info = client.get_meta()
            logging.info(f"Versao do Weaviate: {info.get('version', 'N/A')}")
        logging.info("Conexao fechada automaticamente.")
    except ConnectionError:
        logging.error(
            "Nao foi possivel conectar ao Weaviate. Verifique se ele esta em execucao."
        )
    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")
