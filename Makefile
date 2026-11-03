.PHONY: up down test lint
up:
	docker compose up --build -d
down:
	docker compose down
test:
	pytest
lint:
	ruff check . && ruff format --check .
