# Atlas RAG Engine

Backend of **Atlas**, a multi-workspace RAG platform: hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, a knowledge graph layer, agentic query planning, citation-grounded answers, and a built-in evaluation harness.

> **Status:** early development. Done: typed configuration, app factory, Docker Compose stack, database models and migration (1.1-1.4), pluggable model/vector-store providers (1.5). Next: auth and CRUD APIs (1.6-1.8), then ingestion. Retrieval, agents and evaluation described below are planned, not built.

## Planned architecture

```
Ingest -> Parse -> Chunk -> Embed -> Vector store + BM25 index + Knowledge graph
Query  -> Router -> Rewrite/HyDE -> Hybrid retrieve -> Rerank -> Grounded answer + citations
                 \-> Agent loop (plan, retrieve, reflect) for multi-hop questions
Eval   -> Golden sets, RAGAS-style metrics, regression gate in CI
```

## Roadmap
1. Foundation: config, layout, Docker, database models (done); providers, auth, CRUD (in progress)
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
LLM_PROVIDER=fake EMBED_PROVIDER=fake VECTOR_STORE=memory uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> and try:

1. `GET /v1/providers`: shows the configured LLM, embedder and vector store (never keys).
2. `POST /v1/providers/playground/chat`: send `{"prompt": "Say hello"}`; the fake LLM echoes it back.
3. `POST /v1/providers/playground/similarity`: use the pre-filled example. The document about vector databases ranks first for "how do I search by meaning?".

The fake providers are deterministic stand-ins for wiring checks (the fake embedder is word-hash based, not semantic). For real models set `LLM_PROVIDER=ollama` (default) with Ollama running, or `openai` / `anthropic` with an API key; see `.env.example`. The playground endpoints return 404 when `APP_ENV=prod`.

## Providers

| Interface | Implementations |
|---|---|
| `LLM` | Ollama, OpenAI-compatible (OpenAI, vLLM, LM Studio, OpenRouter), Anthropic, fake |
| `Embedder` | Ollama, OpenAI-compatible, fake |
| `Reranker` | fake (cross-encoder arrives in Phase 3.4) |
| `VectorStore` | Qdrant (REST), in-memory |

## License
MIT
