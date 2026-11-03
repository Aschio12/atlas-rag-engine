from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, v1
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestContextMiddleware
from app.providers.registry import Providers


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)
    providers = Providers(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await providers.aclose()

    app = FastAPI(title="Atlas RAG Engine", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.providers = providers

    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(v1.router)
    return app


app = create_app()
