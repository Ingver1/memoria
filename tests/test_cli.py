import json
import os

import httpx
import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from memory_system.cli import app, API_URL_ENV
from memory_system.config.settings import UnifiedSettings


runner = CliRunner()


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport) -> None:
    """Patch the CLI HTTP client to use the given transport."""

    def _client(url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=url, transport=handler)

    monkeypatch.setattr("memory_system.cli._client", _client)


def test_add_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI add should POST to /memory/add and display ID."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/memory/add"
        data = json.loads(request.content.decode())
        assert data["text"] == "hello"
        return httpx.Response(200, json={"id": "123"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["add", "hello", "--url", "http://api"])
    assert result.exit_code == 0
    assert "123" in result.output


def test_search_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI search should GET /memory/search and print results."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/memory/search"
        assert request.url.params["q"] == "foo"
        assert request.url.params["k"] == "2"
        return httpx.Response(200, json=[{"text": "foo result", "score": 0.9}])

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["search", "foo", "--k", "2", "--url", "http://api"])
    assert result.exit_code == 0
    assert "foo result" in result.output


def test_delete_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI delete should issue DELETE /memory/{id}."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/memory/xyz"
        return httpx.Response(200, json={"status": "deleted"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["delete", "xyz", "--url", "http://api"])
    assert result.exit_code == 0
    assert "Deleted" in result.output


def test_cli_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base URL should come from environment when --url not given."""
    base_url = "http://env-api"
    monkeypatch.setenv(API_URL_ENV, base_url)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(base_url)
        return httpx.Response(200, json={"id": "env"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["add", "ping"])
    assert result.exit_code == 0
    assert "env" in result.output


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """UnifiedSettings should respect AI_ environment variables."""
    monkeypatch.setenv("AI_SECURITY__API_TOKEN", "token-override")

    settings = UnifiedSettings()
    assert settings.security.api_token == "token-override"
