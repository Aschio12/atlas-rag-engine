from fastapi import FastAPI

app = FastAPI(title="Atlas RAG Engine", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "atlas-rag-engine"}
