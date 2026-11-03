from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.deps import SettingsDep
from app.main import create_app


def test_health():
    r = TestClient(create_app(Settings(_env_file=None))).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "atlas-rag-engine"}


def test_settings_dependency_uses_app_settings():
    app: FastAPI = create_app(Settings(_env_file=None, app_env="test"))

    @app.get("/_env")
    def env(s: SettingsDep) -> dict[str, str]:
        return {"env": s.app_env}

    assert TestClient(app).get("/_env").json() == {"env": "test"}
