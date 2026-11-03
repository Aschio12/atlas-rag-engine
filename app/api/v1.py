"""Version 1 router. Feature routers are mounted here as they are built."""

from fastapi import APIRouter

from app.api import auth, providers, workspaces

router = APIRouter(prefix="/v1")
router.include_router(providers.router)
router.include_router(auth.router)
router.include_router(workspaces.router)
