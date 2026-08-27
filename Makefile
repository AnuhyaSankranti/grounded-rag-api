.PHONY: install run demo test lint eval load docker

install:
	python -m pip install -e ".[dev]"

run:
	uvicorn mini_sia.main:app --reload --port 8000

demo:
	MINI_SIA_LLM_PROVIDER=extractive MINI_SIA_EMBEDDING_PROVIDER=hash uvicorn mini_sia.main:app --reload --port 8000

test:
	pytest --cov=mini_sia --cov-report=term-missing

lint:
	ruff check .

eval:
	mini-sia-eval --local --fail-under 0.70

load:
	locust -f loadtests/locustfile.py --host http://localhost:8000

docker:
	docker compose up --build

