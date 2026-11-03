from fastapi import FastAPI

from app.api import health, v1
from app.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Atlas RAG Engine", version="0.1.0")
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(v1.router)
    return app


app = create_app()
