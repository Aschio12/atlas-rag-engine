"""Version 1 router. Feature routers are mounted here as they are built."""

from fastapi import APIRouter

from app.api import providers

router = APIRouter(prefix="/v1")
router.include_router(providers.router)
