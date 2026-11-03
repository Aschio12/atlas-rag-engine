"""Provider status and a playground for checking the wiring from a browser (/docs)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.deps import SettingsDep
from app.providers.base import ChatMessage, Point
from app.providers.registry import Providers

router = APIRouter(prefix="/providers", tags=["providers"])


def providers_dep(request: Request) -> Providers:
    return request.app.state.providers


ProvidersDep = Annotated[Providers, Depends(providers_dep)]


def playground_enabled(settings: SettingsDep) -> None:
    if settings.app_env == "prod":
        raise HTTPException(404, "Not found")


Playground = Depends(playground_enabled)


@router.get("")
def status(settings: SettingsDep) -> dict:
    """Which providers are configured. Never returns keys."""
    return {
        "app_env": settings.app_env,
        "llm": {"provider": settings.llm_provider, "model": settings.resolved_llm_model},
        "embedder": {"provider": settings.embed_provider, "model": settings.resolved_embed_model},
        "vector_store": {"provider": settings.vector_store},
        "playground_enabled": settings.app_env != "prod",
    }


class ChatIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000, examples=["Say hello in five words."])
    system: str | None = Field(default=None, max_length=2000)


@router.post("/playground/chat", dependencies=[Playground])
async def playground_chat(body: ChatIn, providers: ProvidersDep) -> dict:
    """Send one prompt to the configured LLM."""
    messages = [ChatMessage("user", body.prompt)]
    if body.system:
        messages.insert(0, ChatMessage("system", body.system))
    r = await providers.llm.complete(messages, max_tokens=256)
    return {
        "text": r.text,
        "model": r.model,
        "usage": {
            "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens,
        },
    }


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000, examples=["how do I search by meaning?"])
    documents: list[str] = Field(
        min_length=1,
        max_length=50,
        examples=[
            [
                "Vector databases find text by meaning using embeddings.",
                "Chocolate cake needs flour, sugar and cocoa.",
                "FastAPI is a Python web framework.",
            ]
        ],
    )
    limit: int = Field(default=3, ge=1, le=10)


@router.post("/playground/similarity", dependencies=[Playground])
async def playground_similarity(body: SearchIn, providers: ProvidersDep) -> dict:
    """Embed the documents, store them in a throwaway collection, and rank them for the query.

    Exercises the embedder and the vector store together. With the Qdrant store, a temporary
    collection is created and removed again.
    """
    embedder, store = providers.embedder, providers.vector_store
    vectors = await embedder.embed([*body.documents, body.query])
    *doc_vecs, query_vec = vectors
    collection = f"playground-{uuid.uuid4().hex[:12]}"
    await store.ensure_collection(collection, len(query_vec))
    try:
        points = [
            Point(str(uuid.uuid4()), v, {"text": t})
            for t, v in zip(body.documents, doc_vecs, strict=True)
        ]
        await store.upsert(collection, points)
        hits = await store.search(collection, query_vec, limit=body.limit)
    finally:
        await store.drop_collection(collection)
    return {
        "dimension": len(query_vec),
        "embedder": f"{embedder.name}:{embedder.model}",
        "store": store.name,
        "results": [{"text": h.payload["text"], "score": round(h.score, 4)} for h in hits],
    }
