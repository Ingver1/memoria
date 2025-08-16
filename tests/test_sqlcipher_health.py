from pathlib import Path
from urllib.parse import quote

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

import memory_system.settings as settings_module
from memory_system.api.app import create_app
from memory_system.settings import DatabaseConfig, SecurityConfig, UnifiedSettings

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx, pytest.mark.needs_crypto]


def _cfg(tmp_path: Path, monkeypatch: MonkeyPatch, fernet_key: str) -> UnifiedSettings:
    monkeypatch.setattr(settings_module, "_CRYPTO_OK", True)
    return UnifiedSettings.for_testing().model_copy(
        update={
            "database": DatabaseConfig(db_path=tmp_path / "memory.db"),
            "security": SecurityConfig(
                encrypt_at_rest=True,
                encryption_key=fernet_key,
                api_token="test-token-12345678",
                rate_limit_per_minute=10_000,
            ),
        }
    )


def test_bad_key_returns_503(tmp_path: Path, monkeypatch: MonkeyPatch, fernet_key: str) -> None:
    cfg = _cfg(tmp_path, monkeypatch, fernet_key)
    cfg.database.db_path.write_bytes(b"not a database")
    app = create_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["healthy"] is False


def test_missing_database_returns_503(
    tmp_path: Path, monkeypatch: MonkeyPatch, fernet_key: str
) -> None:
    cfg = _cfg(tmp_path, monkeypatch, fernet_key)
    db_path = cfg.database.db_path
    if db_path.exists():
        db_path.unlink()

    def bad_url() -> str:
        return f"sqlite+sqlcipher:///{db_path}?cipher_secret={quote(cfg.security.encryption_key.get_secret_value())}&mode=rw"

    monkeypatch.setattr(cfg, "get_database_url", bad_url)
    app = create_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["healthy"] is False
