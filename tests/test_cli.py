import json
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from memory_system.cli import API_URL_ENV, app
from memory_system.settings import UnifiedSettings

runner = CliRunner()

pytestmark = pytest.mark.needs_httpx


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
        assert request.headers.get("Idempotency-Key")
        data = json.loads(request.content.decode())
        assert data["text"] == "hello"
        return httpx.Response(200, json={"id": "123"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["add", "hello", "--url", "http://api"])
    assert result.exit_code == 0
    assert "123" in result.output


def test_metadata_option_requires_object() -> None:
    """Non-object metadata JSON should raise a parameter error."""
    result = runner.invoke(app, ["add", "hello", "--metadata", "[]"])
    assert result.exit_code != 0
    assert "JSON object" in result.output


def test_search_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI search should POST /memory/search and print results."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/memory/search"
        data = json.loads(request.content.decode())
        assert data["query"] == "foo"
        assert data["top_k"] == 2
        assert data["modality"] == "image"
        assert data["metadata_filter"] == {"tag": "x"}
        assert data["level"] == 1
        return httpx.Response(200, json=[{"text": "foo result", "score": 0.9}])

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(
        app,
        [
            "search",
            "foo",
            "--top-k",
            "2",
            "--modality",
            "image",
            "--metadata-filter",
            '{"tag": "x"}',
            "--level",
            "1",
            "--url",
            "http://api",
        ],
    )
    assert result.exit_code == 0
    assert "foo result" in result.output


def test_consolidate_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI consolidate should POST to /memory/consolidate."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/memory/consolidate"
        assert request.headers.get("Idempotency-Key")
        return httpx.Response(200, json={"created": 3})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["consolidate", "--url", "http://api"])
    assert result.exit_code == 0
    assert "3" in result.output


def test_forget_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI forget should POST to /memory/forget."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/memory/forget"
        assert request.headers.get("Idempotency-Key")
        return httpx.Response(200, json={"deleted": 2})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["forget", "--url", "http://api"])
    assert result.exit_code == 0
    assert "2" in result.output


def test_add_retry_preserves_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying add uses the same Idempotency-Key header."""

    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Idempotency-Key"))
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("boom")
        return httpx.Response(200, json={"id": "321"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(
        app,
        ["add", "hello", "--url", "http://api", "--retry", "1", "--key", "same"],
    )
    assert result.exit_code == 0
    assert calls == ["same", "same"]


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


def test_import_json_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI import-json should POST each line to /memory/add."""
    file = tmp_path / "data.jsonl"
    file.write_text('{"text": "foo"}\n{"text": "bar"}\n')

    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/memory/add"
        posted.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"id": "ok"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["import-json", str(file), "--url", "http://api"])
    assert result.exit_code == 0
    assert "Imported" in result.output
    assert len(posted) == 2
    assert posted[0]["text"] == "foo"
    assert posted[1]["text"] == "bar"


def test_import_json_large_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing a large JSONL file should process all entries."""
    file = tmp_path / "big.jsonl"
    lines = "\n".join(json.dumps({"text": f"item{i}"}) for i in range(50))
    file.write_text(lines)

    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"id": "ok"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    result = runner.invoke(app, ["import-json", str(file), "--url", "http://api"])
    assert result.exit_code == 0
    assert len(posted) == 50


def test_import_json_async_large_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Large imports should use async file I/O instead of blocking Path.open."""
    file = tmp_path / "huge.jsonl"
    lines = "\n".join(json.dumps({"text": f"item{i}"}) for i in range(100))
    file.write_text(lines)

    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"id": "ok"})

    transport = httpx.MockTransport(handler)
    _patch_client(monkeypatch, transport)

    # If the importer uses Path.open (synchronous I/O) this will raise
    def _sync_open(*_: Any, **__: Any) -> Any:  # pragma: no cover - guard
        msg = "synchronous open should not be used"
        raise AssertionError(msg)

    monkeypatch.setattr(Path, "open", _sync_open)

    result = runner.invoke(app, ["import-json", str(file), "--url", "http://api"])
    assert result.exit_code == 0
    assert len(posted) == 100
