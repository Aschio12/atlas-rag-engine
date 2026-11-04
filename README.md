# Atlas RAG Engine

Backend of **Atlas**, a multi-workspace RAG platform: hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, a knowledge graph layer, agentic query planning, citation-grounded answers, and a built-in evaluation harness.

> **Status:** early development. **Phases 1 and 2 are complete**: configuration, Docker stack, database and migrations, pluggable providers, JWT auth with workspace roles, structured logging and Problem+JSON errors, workspace/collection CRUD, and the whole **ingestion pipeline**: upload / text / URL / website / GitHub sources, a durable job queue with live progress, parsers (PDF, DOCX, PPTX, HTML, Markdown, CSV, text, code; OCR fallback), five chunking strategies, enrichment, dense + sparse vector indexing, re-indexing and complete deletion. **Phase 3 (retrieval) has started:** dense search with metadata filters is built (3.1); sparse search, fusion, reranking, query transforms and context assembly come next. Generation, agents and evaluation described below are planned, not built.

## Planned architecture

```
Ingest -> Parse -> Chunk -> Embed -> Vector store + BM25 index + Knowledge graph
Query  -> Router -> Rewrite/HyDE -> Hybrid retrieve -> Rerank -> Grounded answer + citations
                 \-> Agent loop (plan, retrieve, reflect) for multi-hop questions
Eval   -> Golden sets, RAGAS-style metrics, regression gate in CI
```

## Roadmap
1. Foundation: config, Docker, database, providers, auth, errors/logging, CRUD (done)
2. Ingestion: upload, queue, parsers, chunking, enrichment, indexing, connectors, re-indexing (done)
3. Hybrid retrieval and reranking (in progress: 3.1 dense search done)
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
10. **Choose how documents are split**: when creating a collection set `chunking_strategy` (`fixed`, `recursive`, `semantic`, `parent_child`, `contextual`) and optionally `chunking_config`. Try `POST /v1/workspaces/{workspace_id}/chunking/preview` first: paste text, pick a strategy and sizes and see the chunks and statistics without storing anything. After a document is ready, `GET .../documents/{document_id}/chunks` lists its stored chunks with offsets, page and heading breadcrumb.
11. **Enrich and index**: every ready document has `metadata.indexing` (points written, vector size, model). `GET .../collections/{collection_id}/index` compares the vector store's point count with what the documents wrote (`in_sync`). Create a collection with `"enrichment": {"summaries": true, "questions": true}` to add LLM summaries and hypothetical questions to each chunk (they show up in `.../chunks`).
12. **Connect a source**: `POST .../collections/{collection_id}/sources` with `{"type": "web", "config": {"url": "https://example.com/docs/"}}` (reads the sitemap) or `{"type": "github", "config": {"repo": "owner/name"}}`. It returns 202 right away; poll `GET .../sources/{source_id}` for `status` and counters (`created`, `updated`, `unchanged`, `removed`, `failed`). `POST .../sync` runs it again: only changed items are re-ingested and vanished ones are removed.
13. **Change your mind**: `POST .../collections/{collection_id}/reindex` with a new `chunking_strategy` / `chunking_config` / `enrichment` / `embed_model` rebuilds every document from its stored text (admin only; refused while documents are still processing). `POST .../documents/{document_id}/reindex` rebuilds one.
14. **Search**: `POST /v1/workspaces/{workspace_id}/search` with `{"query": "how long do refunds take?"}` returns the best passages with their page, heading path, character offsets, scores and the reasons they were chosen, plus a `trace` of each stage. Narrow it with `collection_ids` or `filters` (for example `[{"field": "language", "op": "eq", "value": "en"}, {"field": "created_at", "op": "gte", "value": "2024-01-01"}]`); `GET .../search/fields` lists what can be filtered on.
15. **Try the guards**: upload the same file again (409 with the existing document id), rename an `.exe` to `.pdf` (415), upload something over the size limit (413), submit `http://169.254.169.254/` as a URL (400, blocked).

The fake providers are deterministic stand-ins for wiring checks. For real models set `LLM_PROVIDER=ollama` (default) with Ollama running, or `openai` / `anthropic` with an API key; see `.env.example`. Playground endpoints return 404 when `APP_ENV=prod`.

## API

| Area | Endpoints | Access |
|---|---|---|
| Auth | `POST /v1/auth/register`, `POST /v1/auth/login`, `GET /v1/auth/me` | public / token |
| Workspaces | `POST,GET /v1/workspaces`, `GET,PATCH,DELETE /v1/workspaces/{id}` | any user / viewer / admin / owner |
| Collections | `POST,GET /v1/workspaces/{id}/collections`, `GET,PATCH,DELETE .../{collection_id}` | member (write), viewer (read), admin (delete) |
| Documents | `POST /v1/workspaces/{id}/collections/{cid}/documents` (file), `.../documents/text`, `.../documents/url`; `GET` list / detail / `.../parsed` / `.../original`; `DELETE` | member (write), viewer (read) |
| Chunks | `GET .../documents/{document_id}/chunks`, `POST /v1/workspaces/{id}/chunking/preview` | viewer (read), member (preview) |
| Sources | `POST,GET .../collections/{cid}/sources`, `GET,DELETE .../sources/{id}`, `POST .../sources/{id}/sync` | member (write), viewer (read) |
| Index | `GET .../collections/{cid}/index`, `POST .../collections/{cid}/reindex`, `POST .../documents/{did}/reindex` | viewer (status), admin (collection) / member (document) |
| Search | `POST /v1/workspaces/{id}/search`, `GET /v1/workspaces/{id}/search/fields` | viewer |
| Jobs | `GET /v1/workspaces/{id}/jobs`, `GET .../jobs/{job_id}`, `GET .../jobs/{job_id}/events` (SSE), `POST .../jobs/{job_id}/retry` | viewer (read), member (retry) |

Roles are `viewer < member < admin < owner`. Non-members receive 404, never 403, so workspace ids cannot be probed.

## Ingestion (Phase 2)

```
upload / text / URL / website / GitHub
        └─> validate + dedupe ─> blob store ─> job queue ─> worker
              parse ─> chunk ─> enrich ─> embed ─> index ─> ready
                                           │          └─ verify, then drop the previous generation
                                           └─ dense + sparse vectors, payload, question points
```

* **Safe by default.** File type is decided from the bytes (an `.exe` renamed `.pdf` is rejected), size is capped while reading, identical content in a collection is rejected with the existing document id, originals are served only as attachments, and ZIP/XML bombs and XXE are blocked for Office files.
* **URL ingestion is SSRF-guarded.** Only http(s); every resolved address must be public (no loopback, private, link-local or cloud-metadata ranges); redirects are re-checked at each hop; body size and time are capped. Set `ALLOW_PRIVATE_URLS=true` only for local development. (Known limit: validate-then-connect, so DNS rebinding by a hostile resolver is not fully closed; put a filtering egress proxy in front for hostile multi-tenant use.)
* **Durable queue, no extra service.** Jobs live in the database with a lease: a crashed worker's job is picked up again, retries use exponential backoff, and a permanently bad file fails fast with a message you can show a user. It works on SQLite and Postgres and is safe with several workers. Stages report progress, so the live event stream moves smoothly through embedding.
* **Parsers that keep structure.** Headings, page/slide numbers, lists, tables (`cell | cell` rows), PDF multi-column reading order, hyphenation repair, removal of repeated page headers/footers, speaker notes, CSV rows labelled by column, source code with its language. Scanned PDFs use Tesseract if installed (`pip install .[ocr]` plus the `tesseract` binary, or `docker build --build-arg WITH_OCR=true .`); otherwise those pages are skipped with a warning.
* **One coordinate system.** All offsets (headings, pages, every chunk and every future citation) index the single normalised text returned by `.../parsed`.

### Enrichment (2.5)

Each searchable chunk gets metadata stored with it and returned by the chunk API:

| Field | How | Cost |
|---|---|---|
| `language` | Script detection plus function-word profiles for ten Latin-script languages; no guess beats a wrong guess | free, offline |
| `keywords` | TF-IDF across the document's own chunks, phrases preferred, stopwords and numbers excluded | free, offline |
| `summary` | LLM, one sentence (opt-in `summaries`) | one LLM call per chunk |
| `questions` | LLM, questions the chunk answers (opt-in `questions`); each becomes its own vector so user questions match them directly | same call |

LLM enrichment has bounded concurrency, strict JSON validation (junk, duplicates and non-questions are dropped), a per-document cap (`max_chunks`), per-chunk failure tolerance, and retries when the provider is down.

### Embedding and indexing (2.6)

Each chunk is one vector-store point whose id is the chunk id: a **dense** vector from the collection's own embedding model plus a **sparse** BM25-style vector (term-frequency weights, accent-folded and lightly stemmed; Qdrant supplies inverse document frequency with its `idf` modifier). The payload carries workspace, collection, document, chunk, parent, page, character offsets, heading path, language and title, so retrieval can filter by tenant and cite exactly. Since Phase 3.1 it also carries the document's source type, source id, file type and creation time (Unix seconds); documents indexed before that lack these fields until re-indexed (`POST .../reindex`), and filters on them skip such documents. Hypothetical questions are extra points that lead back to their chunk; parent chunks (parent-child) are stored but not indexed. Text is not duplicated into the vector store: the database stays the source of truth.

Every indexing run has a fresh `index_version`. The index stage verifies the run is complete and only then deletes the document's older points, so a document never disappears from search while it is re-indexed, and a retry cannot duplicate anything.

### Sources (2.7)

* **Website**: `sitemap` mode reads `sitemap.xml` (indexes, gzip, `robots.txt` hints) and `links` mode follows in-scope links to a depth limit. It stays on the starting host and folder, honours include/exclude globs, **always obeys robots.txt** (including Crawl-delay; a forbidden or failing robots.txt means no crawl), paces requests, caps pages (`max_pages`, at most 500) and uses the SSRF-safe fetcher.
* **GitHub**: docs and source files of a repository, pinned to one commit, with vendored/built/lock files skipped, folder/extension filters, a file cap that keeps README and docs first, and blob-SHA checks so unchanged files are not downloaded again. `GITHUB_TOKEN` (optional) unlocks private repositories and a higher rate limit.
* **Re-sync**: new items are added, changed ones are updated in place and re-ingested, unchanged ones cost nothing, and items that vanished are removed (with their vectors and files), but only after a complete, failure-free listing. Not built: Google Drive, Notion, Slack exports (stretch goals).

### Re-indexing and deletion (2.8)

* Changing a collection's strategy, chunk sizes, enrichment or embedding model is only possible through `POST .../reindex`, which saves the settings and rebuilds every document from its stored text. Old vectors stay searchable until each document's new ones are complete; only an embedding-model change drops the index first (vectors from two models cannot share one), so search for that collection is partial until the jobs finish. Watch `GET .../index`.
* Deleting a document, collection, source (with `delete_documents=true`) or workspace removes its vectors first, then its rows, then its files. If the vector store is down the delete fails with a 502 and nothing is orphaned. `app.ingest.purge` has hooks so the Phase 5 knowledge graph can register its own cleanup.

### Tests for ingestion (2.9)

Beyond unit and API tests: **golden snapshots** (`tests/golden`) pin the exact parsed text, headings, pages, metadata and chunk spans for eleven committed input files (real DOCX/PPTX/PDF bytes, messy HTML, CSV, cp1252 and UTF-16 text, Unicode/RTL/emoji Markdown, code); **property and fuzz tests** check that every chunk of every strategy is an exact slice, nothing is lost, sizes are bounded and breadcrumbs are right, on real formats, random bytes, truncated and bit-flipped files; and `tests/integration` runs the Qdrant adapter against a real server in CI. Refresh snapshots after an intentional change with `python tests/golden/regenerate.py` and review the diff.

Limitations: DOCX has no page numbers (Word paginates at render time); text boxes and footnotes in DOCX are not read; PDF tables without ruling lines are read as text. The Qdrant adapter is verified against a real server only in CI, not on the author's machine (Docker Hub was unreachable there).

### Chunking strategies (2.4)

Every chunker works on character spans of the single parsed text, so each chunk is an exact slice of it (`text[char_start:char_end]`): citations can highlight the precise passage and no text is ever lost or rewritten. Sizes are estimated tokens and set per collection (`target_tokens`, `overlap_tokens`, `max_tokens`, ...).

| Strategy | What it does | Use it when | Cost |
|---|---|---|---|
| `fixed` | Sliding window with overlap; cuts snap to whitespace | A baseline to compare against | none |
| `recursive` (default) | Splits by section, paragraph, line, sentence, word; merges up to the target; keeps a heading with its text, never strands one, overlaps whole sentences without crossing sections, keeps table rows and list items whole | Most documents | none |
| `semantic` | Embeds sentences and cuts where the topic shifts (distance above a percentile), within size limits | Long mixed-topic text | one embedding per sentence |
| `parent_child` | Small child chunks are searched; each points at a large parent chunk used as answer context | Answers that need surrounding context | none (more rows) |
| `contextual` | An LLM writes a short situating sentence per chunk, embedded with the passage; the document is a cached prompt prefix, calls are capped by `context_max_chunks`, failures degrade gracefully | Ambiguous chunks ("revenue grew 3%" of whom?) | one LLM call per chunk (cached prefix) |

Which one is best on *your* data is an empirical question: Phase 6 adds an evaluation harness that will report the numbers.

## Retrieval (Phase 3)

### Dense search (3.1)

`POST /v1/workspaces/{id}/search` embeds the query once per embedding model in play, searches every collection of the workspace (or the `collection_ids` you name) in the vector store, and returns up to `top_k` (1 to 50) passages.

* **Always scoped.** Workspace and collection filters are added by the server after yours, and the filter language cannot name them. A `collection_ids` entry from another workspace gets the same 404 as one that does not exist. Viewers can search.
* **Metadata filters** are a list of conditions that must all hold, on an allow-listed set of fields: `document_id`, `source_id`, `source_type`, `file_type`, `language`, `title`, `heading`, `page`, `created_at`. Operators are `eq`, `ne`, `in`, `nin` and, for `page` and `created_at`, `gt`, `gte`, `lt`, `lte`. Codes are lowercased, ids validated, dates parsed (ISO 8601, UTC when no zone is given), and a bare date means the whole day: `lte 2023-12-31` includes the 31st. Redundant or contradictory combinations are a 422 with a readable message instead of a silent guess. They run inside the vector store, so top-k is taken after filtering.
* **What is served is current.** Passage text comes from the database. A point is used only if its chunk exists in the collection it claims and its `index_version` is the one the document's record says is current, so leftovers of an interrupted or older indexing run never show up, nor hide valid results. Then points are collapsed per chunk: a chunk reached through one of its generated questions is returned once, with `matched: "question"` and the question that matched.
* **Explained.** Each hit has `scores` per stage (`{"dense": 0.83}`) and `reasons`; `trace.stages` records scope, query embedding, search and resolution with timings and counts (points found, dropped as missing or stale, folded, below `min_score`). When collections use different embedding models the trace says their scores are not comparable; the merged order then follows raw similarity.
* **Limits.** Candidates are fetched at three times `top_k` (at least 10, at most 200) per collection to survive dropped points, so a search that has to discard most of them can return fewer than `top_k` hits even though more exist. Query text is capped at 2000 characters. The `mode` field only accepts `dense` for now; sparse and hybrid arrive in 3.2 and 3.3.

## Providers

| Interface | Implementations |
|---|---|
| `LLM` | Ollama, OpenAI-compatible (OpenAI, vLLM, LM Studio, OpenRouter), Anthropic, fake |
| `Embedder` | Ollama, OpenAI-compatible, fake |
| `Reranker` | fake (cross-encoder arrives in Phase 3.4) |
| `VectorStore` | Qdrant (REST), in-memory |

## License
MIT
