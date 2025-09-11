# Configurações
IMAGE_NAME=flask-gunicorn-app
PORT ?= 5000
ENV_FILE=.env
VENV=.venv

requirements:
	@pip freeze > requirements.txt

venv:
	@python3.11 -m venv $(VENV)
	@/bin/zsh -i -c "source $(VENV)/bin/activate"

install:
	@$(VENV)/bin/pip install -r requirements.txt

wsgi:
	@$(VENV)/bin/gunicorn wsgi:app --bind 0.0.0.0:$(PORT)

weaviate:
	@docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=20 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e ENABLE_GRPC=true \
  -e WEAVIATE_DEFAULT_VECTORIZER=text2vec-transformers \
  -e WEAVIATE_ENABLE_MODULES=text2vec-transformers \
  --name weaviate \
  semitechnologies/weaviate:latest


bot:
	@streamlit run app/agent/main.py