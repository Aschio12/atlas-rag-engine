# Atlas RAG Engine

Backend of **Atlas**, a multi-workspace RAG platform: hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, a knowledge graph layer, agentic query planning, citation-grounded answers, and a built-in evaluation harness.

> **Status:** early development. Done: typed configuration, app factory, Docker Compose stack, database models and the initial migration (Phase 1.1-1.4). Next: auth and CRUD APIs (1.5-1.8), then ingestion. Retrieval, agents and evaluation described below are planned, not built.

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

## License
MIT
