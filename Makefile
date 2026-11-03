.PHONY: up down logs ps infra seed test lint fmt migrate

up:
	docker compose up --build -d
down:
	docker compose down
logs:
	docker compose logs -f api
ps:
	docker compose ps
# Start only the data services (lowest RAM): run the API on the host with uvicorn.
infra:
	docker compose up -d postgres qdrant redis
migrate:
	alembic upgrade head
seed:
	@echo "Sample corpus loading arrives with Phase 2 (ingestion)."
test:
	pytest
lint:
	ruff check . && ruff format --check .
fmt:
	ruff check --fix . && ruff format .
