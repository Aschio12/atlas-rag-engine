"""Version 1 router. Feature routers are mounted here as they are built."""

from fastapi import APIRouter

from app.api import (
    answer,
    auth,
    chunks,
    collections,
    conversations,
    documents,
    jobs,
    memory,
    policy,
    providers,
    reindex,
    retrievals,
    search,
    sources,
    workspaces,
)

router = APIRouter(prefix="/v1")
router.include_router(providers.router)
router.include_router(auth.router)
router.include_router(workspaces.router)
router.include_router(collections.router)
router.include_router(documents.router)
router.include_router(jobs.router)
router.include_router(chunks.router)
router.include_router(reindex.router)
router.include_router(sources.router)
router.include_router(search.router)
router.include_router(retrievals.router)
router.include_router(answer.router)
router.include_router(policy.router)
router.include_router(memory.router)
router.include_router(conversations.router)
