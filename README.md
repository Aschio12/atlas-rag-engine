# Atlas RAG Engine

Backend of **Atlas**, a multi-workspace RAG platform: hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, a knowledge graph layer, agentic query planning, citation-grounded answers, and a built-in evaluation harness.

> **Status:** early development. Done: typed configuration, app factory, Docker Compose stack, database models and migration (1.1-1.4), pluggable providers (1.5), JWT auth with workspace roles (1.6), structured logging and Problem+JSON errors (1.7), workspace and collection CRUD (1.8). Phase 1 is complete; next is ingestion (Phase 2), then ingestion. Retrieval, agents and evaluation described below are planned, not built.

## Planned architecture

```
Ingest -> Parse -> Chunk -> Embed -> Vector store + BM25 index + Knowledge graph
Query  -> Router -> Rewrite/HyDE -> Hybrid retrieve -> Rerank -> Grounded answer + citations
                 \-> Agent loop (plan, retrieve, reflect) for multi-hop questions
Eval   -> Golden sets, RAGAS-style metrics, regression gate in CI
```

## Roadmap
1. Foundation: config, Docker, database, providers, auth, errors/logging, CRUD (done)
2. Ingestion and chunking
3. Hybrid retrieval and reranking
4. Grounded generation with citations
5. Agentic RAG and graph retrieval
6. Evaluation and observability

Companion UI: [atlas-rag-studio](https://github.com/Aschio12/atlas-rag-studio).

## Data model

`users`, `workspaces`, `workspace_members`, `collections`, `documents`, `chunks` (with character offsets for citation highlighting), `ingest_jobs`, `conversations`, `messages` (citations, retrieval trace, usage), `eval_runs`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make infra            # postgres, qdrant, redis only (low RAM)
alembic upgrade head  # create the schema
uvicorn app.main:app --reload
pytest
```

## Try it in your browser (no models, no Docker)

```bash
export DATABASE_URL=sqlite+aiosqlite:///./atlas-dev.db
alembic upgrade head      # creates the local SQLite database
LLM_PROVIDER=fake EMBED_PROVIDER=fake VECTOR_STORE=memory LOG_FORMAT=console \
  uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> (Swagger UI, a testing page for the API, not the product UI) and:

1. **Register**: `POST /v1/auth/register` with the pre-filled example.
2. **Login**: `POST /v1/auth/login` with the same email and password. Copy the `access_token`.
3. **Authorize**: click the green **Authorize** button at the top, paste the token, confirm. Protected endpoints now work.
4. **Workspace**: `POST /v1/workspaces` with `{"name": "Acme Support"}`. Copy its `id`.
5. **Collection**: `POST /v1/workspaces/{workspace_id}/collections` with the workspace id and `{"name": "Product manuals"}`; then `GET` the same path to list it.
6. **Providers**: `POST /v1/providers/playground/chat` and `/similarity` check the model wiring (fake providers echo and rank by shared words).

Errors come back as `application/problem+json` and every response carries an `X-Request-ID` header that also appears in the server log. Try a wrong password (401), a duplicate email (409) or a blank name (422).

The fake providers are deterministic stand-ins for wiring checks. For real models set `LLM_PROVIDER=ollama` (default) with Ollama running, or `openai` / `anthropic` with an API key; see `.env.example`. Playground endpoints return 404 when `APP_ENV=prod`.

## API (Phase 1)

| Area | Endpoints | Access |
|---|---|---|
| Auth | `POST /v1/auth/register`, `POST /v1/auth/login`, `GET /v1/auth/me` | public / token |
| Workspaces | `POST,GET /v1/workspaces`, `GET,PATCH,DELETE /v1/workspaces/{id}` | any user / viewer / admin / owner |
| Collections | `POST,GET /v1/workspaces/{id}/collections`, `GET,PATCH,DELETE .../{collection_id}` | member (write), viewer (read), admin (delete) |

Roles are `viewer < member < admin < owner`. Non-members receive 404, never 403, so workspace ids cannot be probed.

## Providers

| Interface | Implementations |
|---|---|
| `LLM` | Ollama, OpenAI-compatible (OpenAI, vLLM, LM Studio, OpenRouter), Anthropic, fake |
| `Embedder` | Ollama, OpenAI-compatible, fake |
| `Reranker` | fake (cross-encoder arrives in Phase 3.4) |
| `VectorStore` | Qdrant (REST), in-memory |

## License
MIT
