from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import health, v1
from app.core.config import Settings, get_settings
from app.providers.errors import ProviderError
from app.providers.registry import Providers


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    providers = Providers(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await providers.aclose()

    app = FastAPI(title="Atlas RAG Engine", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.providers = providers

    @app.exception_handler(ProviderError)
    async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
        # 502: our upstream (model/vector DB) failed, not the caller's request.
        return JSONResponse(
            status_code=502,
            content={"error": "provider_error", "provider": exc.provider, "detail": str(exc)},
        )

    app.include_router(health.router)
    app.include_router(v1.router)
    return app


app = create_app()
