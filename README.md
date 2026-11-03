# Atlas RAG Engine

Backend of **Atlas**, a multi-workspace RAG platform: hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, a knowledge graph layer, agentic query planning, citation-grounded answers, and a built-in evaluation harness.

> **Status:** early development. Only the service skeleton and `/health` exist today; the roadmap below is what is being built, in order.

## Planned architecture

```
Ingest -> Parse -> Chunk -> Embed -> Vector store + BM25 index + Knowledge graph
Query  -> Router -> Rewrite/HyDE -> Hybrid retrieve -> Rerank -> Grounded answer + citations
                 \-> Agent loop (plan, retrieve, reflect) for multi-hop questions
Eval   -> Golden sets, RAGAS-style metrics, regression gate in CI
```

## Roadmap
1. Foundation, config, health, CI (current)
2. Ingestion and chunking
3. Hybrid retrieval and reranking
4. Grounded generation with citations
5. Agentic RAG and graph retrieval
6. Evaluation and observability

Companion UI: [atlas-rag-studio](https://github.com/Aschio12/atlas-rag-studio).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest
```

## License
MIT
