# Atlas RAG Engine

Backend of **Atlas**, a multi-workspace RAG platform: hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, a knowledge graph layer, agentic query planning, citation-grounded answers, and a built-in evaluation harness.

> **Status:** early development. Done: configuration, app factory, Docker stack, database models and migrations (1.1-1.4), pluggable providers (1.5), JWT auth with workspace roles (1.6), structured logging and Problem+JSON errors (1.7), workspace and collection CRUD (1.8), **document upload and URL ingestion (2.1), a durable job queue with live progress (2.2) and parsers for PDF, DOCX, PPTX, HTML, Markdown, CSV and text with OCR fallback (2.3)**. Next: chunking (2.4), enrichment (2.5) and embedding/indexing (2.6). Retrieval, agents and evaluation described below are planned, not built.

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

### Upload and watch a document being processed (Phase 2)

7. **Upload**: `POST /v1/workspaces/{workspace_id}/collections/{collection_id}/documents`, choose any PDF, DOCX, PPTX, HTML, Markdown, CSV or text file (or use `.../documents/text` to paste text, or `.../documents/url` to fetch a page). The response already contains a `job`.
8. **Watch the job**: `GET /v1/workspaces/{workspace_id}/jobs/{job_id}` shows `queued` then `done` (or `failed` with a readable reason). `GET .../jobs/{job_id}/events` streams the same changes live (server-sent events).
9. **Read what was extracted**: `GET .../documents/{document_id}/parsed` returns the clean text, the page map and the headings. Every future citation will point into this text.
10. **Try the guards**: upload the same file again (409 with the existing document id), rename an `.exe` to `.pdf` (415), upload something over the size limit (413), submit `http://169.254.169.254/` as a URL (400, blocked).

The fake providers are deterministic stand-ins for wiring checks. For real models set `LLM_PROVIDER=ollama` (default) with Ollama running, or `openai` / `anthropic` with an API key; see `.env.example`. Playground endpoints return 404 when `APP_ENV=prod`.

## API

| Area | Endpoints | Access |
|---|---|---|
| Auth | `POST /v1/auth/register`, `POST /v1/auth/login`, `GET /v1/auth/me` | public / token |
| Workspaces | `POST,GET /v1/workspaces`, `GET,PATCH,DELETE /v1/workspaces/{id}` | any user / viewer / admin / owner |
| Collections | `POST,GET /v1/workspaces/{id}/collections`, `GET,PATCH,DELETE .../{collection_id}` | member (write), viewer (read), admin (delete) |

| Documents | `POST /v1/workspaces/{id}/collections/{cid}/documents` (file), `.../documents/text`, `.../documents/url`; `GET` list / detail / `.../parsed` / `.../original`; `DELETE` | member (write), viewer (read) |
| Jobs | `GET /v1/workspaces/{id}/jobs`, `GET .../jobs/{job_id}`, `GET .../jobs/{job_id}/events` (SSE), `POST .../jobs/{job_id}/retry` | viewer (read), member (retry) |

Roles are `viewer < member < admin < owner`. Non-members receive 404, never 403, so workspace ids cannot be probed.

## Ingestion (Phase 2.1-2.3)

```
upload / text / URL ──> validate + dedupe ──> blob store ──> job queue ──> worker ──> parse ──> parsed.json
```

* **Safe by default.** File type is decided from the bytes (an `.exe` renamed `.pdf` is rejected), size is capped while reading, identical content in a collection is rejected with the existing document id, originals are served only as attachments, and ZIP/XML bombs and XXE are blocked for Office files.
* **URL ingestion is SSRF-guarded.** Only http(s); every resolved address must be public (no loopback, private, link-local or cloud-metadata ranges); redirects are re-checked at each hop; body size and time are capped. Set `ALLOW_PRIVATE_URLS=true` only for local development.
* **Durable queue, no extra service.** Jobs live in the database with a lease: a crashed worker's job is picked up again, retries use exponential backoff, and a permanently bad file fails fast with a message you can show a user. It works on SQLite and Postgres and is safe with several workers.
* **Parsers that keep structure.** Headings, page/slide numbers, lists, tables (`cell | cell` rows), PDF multi-column reading order, hyphenation repair, removal of repeated page headers/footers, speaker notes, CSV rows labelled by column. Scanned PDFs use Tesseract if installed (`pip install .[ocr]` plus the `tesseract` binary, or `docker build --build-arg WITH_OCR=true .`); otherwise those pages are skipped with a warning.
* **One coordinate system.** All offsets (headings, pages, and later every chunk and citation) index the single normalised text returned by `.../parsed`.

Not yet: chunking, embeddings and indexing (2.4-2.6), so `ready` currently means "parsed". Limitations: DOCX has no page numbers (Word paginates at render time); text boxes and footnotes in DOCX are not read; PDF tables without ruling lines are read as text.

## Providers

| Interface | Implementations |
|---|---|
| `LLM` | Ollama, OpenAI-compatible (OpenAI, vLLM, LM Studio, OpenRouter), Anthropic, fake |
| `Embedder` | Ollama, OpenAI-compatible, fake |
| `Reranker` | fake (cross-encoder arrives in Phase 3.4) |
| `VectorStore` | Qdrant (REST), in-memory |

## License
MIT
