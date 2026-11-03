from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, v1
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestContextMiddleware
from app.db.session import make_engine, make_sessionmaker
from app.ingest.fetch import fetch_url
from app.ingest.worker import Worker
from app.providers.registry import Providers
from app.storage import LocalBlobStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)
    providers = Providers(settings)
    engine = make_engine(settings)

    blobs = LocalBlobStore(settings.storage_dir)
    sessionmaker = make_sessionmaker(engine)
    worker = Worker(sessionmaker, settings, blobs, providers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.worker_enabled:
            worker.start()
        yield
        await worker.stop()
        await providers.aclose()
        await engine.dispose()

    app = FastAPI(title="Atlas RAG Engine", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.providers = providers
    app.state.engine = engine
    app.state.blobs = blobs
    app.state.worker = worker
    app.state.fetch_url = fetch_url  # replaced in tests
    app.state.sessionmaker = sessionmaker

    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(v1.router)
    return app


app = create_app()
