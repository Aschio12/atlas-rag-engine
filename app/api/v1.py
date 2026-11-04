"""Version 1 router. Feature routers are mounted here as they are built."""

from fastapi import APIRouter

from app.api import (
    auth,
    chunks,
    collections,
    documents,
    jobs,
    providers,
    reindex,
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
