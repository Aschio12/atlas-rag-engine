"""FastAPI dependency providers (dependency injection entry points)."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings


def settings_dep(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(settings_dep)]
