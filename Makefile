.PHONY: install test api demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

api:
	uvicorn backend.app.main:app --reload --port 8000

demo:
	python -m http.server 5173 --directory demo
