"""
memory_system.cli

Command-line interface for **AI-memory**.

This optional CLI interacts with the running API service via HTTP.  It
lives in the `cli` extra so production deployments can avoid pulling
interactive libraries.  When *rich* is missing we degrade to plain-text
output.
"""

from __future__ import annotations

# ─────────────────────────────── stdlib imports ───────────────────────────────
import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeVar, cast

try:  # pragma: no cover - optional dependency
    import typer as _typer
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully when Typer missing

    class Exit(SystemExit):
        """Fallback ``typer.Exit`` replacement."""

        def __init__(self, code: int = 0) -> None:  # match Typer API
            super().__init__(code)

    class _Typer:
        """Minimal stub so module imports without typer installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def command(self, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
            def decorator(func: Any) -> Any:
                return func

            return decorator

    def _argument(default: Any = None, *args: Any, **kwargs: Any) -> Any:
        return default

    def _option(default: Any = None, *args: Any, **kwargs: Any) -> Any:
        return default

    class _Context:  # pragma: no cover - simple placeholders
        pass

    class _CallbackParam:  # pragma: no cover
        pass

    class _BadParameter(Exception):  # pragma: no cover
        pass

    class _TyperModule:
        Typer = _Typer
        Exit = Exit
        Argument = staticmethod(_argument)
        Option = staticmethod(_option)
        Context = _Context
        CallbackParam = _CallbackParam
        BadParameter = _BadParameter

    typer: Any = _TyperModule()
else:
    typer = _typer

# ────────────────────────────── third-party imports ────────────────────────────
from memory_system.utils.afile import open as aopen
from memory_system.utils.dependencies import require_httpx
from memory_system.utils.http import HTTPTimeouts

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import httpx


Panel: type[Any]
Table: type[Any]

_rich_warning_shown = False
_rich_warning_lock = Lock()

# ---------------------------------------------------------------------------

# Optional rich import (colourful tables & pretty JSON)

# ---------------------------------------------------------------------------

try:
    from rich import print as rprint
    from rich.panel import Panel as RichPanel
    from rich.table import Table as RichTable
except ModuleNotFoundError:  # rich not installed -> degrade gracefully
    from typing import IO

    def rprint(
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        file: IO[str] | None = None,
        flush: bool = False,
    ) -> None:  # fallback printer
        global _rich_warning_shown
        with _rich_warning_lock:
            if not _rich_warning_shown:
                logger.warning("Install extras for colours:  pip install ai-memory[cli]")
                _rich_warning_shown = True
        print(*objects, sep=sep, end=end, file=file, flush=flush)

    class _Panel:
        """Minimal shim so code stays import-safe without *rich*."""

        def __init__(self, renderable: str, **_: Any) -> None:
            self.renderable = renderable

        def __str__(self) -> str:
            return self.renderable

    class _Table:
        """Plain ASCII table shim used when *rich* is missing."""

        def __init__(self, title: str | None = None, **_: Any) -> None:
            self.title = title or ""
            self.rows: list[list[str]] = []

        def add_column(self, *_: Any, **__: Any) -> None:
            return None

        def add_row(self, *values: str) -> None:
            self.rows.append(list(values))

        def __str__(self) -> str:
            head = [self.title] if self.title else []
            lines = [" | ".join(r) for r in self.rows]
            return "\n".join(head + lines)

    Panel = _Panel
    Table = _Table
else:
    Panel = RichPanel
    Table = RichTable

logger = logging.getLogger("memoria.cli")

# ---------------------------------------------------------------------------

# Typer application

# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ai-mem",
    help="Interact with an AI-memory- server via REST API.",
)

API_URL_ENV = "AI_MEM_API_URL"
DEFAULT_API = "http://localhost:8000"

# ---------------------------------------------------------------------------

# Helper utilities

# ---------------------------------------------------------------------------


T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run ``async`` coroutine handling exceptions for CLI commands.

    The coroutine is executed via :func:`asyncio.run` when possible but will
    gracefully fall back to ``asyncio.get_event_loop().run_until_complete`` if
    ``asyncio.run`` cannot be used (for example when an event loop is already
    running).
    """
    try:
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
    except Exception as exc:  # pragma: no cover - simple CLI error path
        rprint(Panel(str(exc), title="Error", style="bold red"))
        raise typer.Exit(1) from exc


def _client(
    base_url: str,
    *,
    timeouts: HTTPTimeouts | None = None,
) -> httpx.AsyncClient:
    """Return shared httpx client with separate connect/read timeouts."""
    hx = require_httpx()
    timeouts = timeouts or HTTPTimeouts(read=30.0)
    timeout = hx.Timeout(read=timeouts.read, connect=timeouts.connect)
    return cast("httpx.AsyncClient", hx.AsyncClient(base_url=base_url, timeout=timeout))


def _metadata_option(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    value: str | None,
) -> dict[str, Any] | None:
    """Parse `--metadata` JSON string into dict."""
    if not value:
        return None
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise typer.BadParameter(f"Invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise typer.BadParameter("Metadata must be a JSON object")
    return result


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retries: int = 0,
    **kwargs: Any,
) -> httpx.Response:
    """Send a request retrying with exponential backoff on failure."""
    attempt = 0
    while True:
        try:
            return await client.request(method, url, **kwargs)
        except Exception as exc:  # pragma: no cover - network errors
            if attempt >= retries:
                raise
            delay = 2**attempt
            logger.warning(
                "%s %s failed (attempt %d/%d): %s",
                method,
                url,
                attempt + 1,
                retries,
                exc,
            )
            await asyncio.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------

# Commands

# ---------------------------------------------------------------------------


@app.command()
def add(
    text: str = typer.Argument(..., help="Text to remember."),
    importance: float = typer.Option(0.5, help="0-1 importance weighting."),
    metadata: str | None = typer.Option(
        None,
        "--metadata",
        callback=_metadata_option,
        help="Arbitrary JSON metadata.",
    ),
    url: str = typer.Option(
        DEFAULT_API,
        "--url",
        envvar=API_URL_ENV,
        show_default="env/localhost",
    ),
    retry: int = typer.Option(0, "--retry", help="Retry attempts on failure."),
    idempotency_key: str | None = typer.Option(
        None,
        "--key",
        "--idempotency-key",
        help="Idempotency key to avoid duplicate submissions.",
    ),
) -> None:
    """Add a new memory row to the store."""

    async def _run_add() -> None:
        async with _client(url) as client:
            payload = {
                "text": text,
                "importance": importance,
                "metadata": metadata or {},
            }
            rprint(f"[grey]POST {url}/memory/add …")
            headers = {"Idempotency-Key": idempotency_key or uuid.uuid4().hex}
            resp = await _request_with_retry(
                client,
                "POST",
                "/memory/add",
                json=payload,
                headers=headers,
                retries=retry,
            )
            resp.raise_for_status()
            rprint(Panel("Memory ID → [bold green]" + resp.json()["id"]))

    run(_run_add())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results."),
    modality: str = typer.Option("text", "--modality", help="Modality of the query."),
    metadata_filter: str | None = typer.Option(
        None,
        "--metadata-filter",
        callback=_metadata_option,
        help="JSON metadata filter.",
    ),
    level: int | None = typer.Option(None, "--level", help="Memory level filter."),
    alpha: float | None = typer.Option(None, "--alpha", help="Weight for vector distance term"),
    beta: float | None = typer.Option(None, "--beta", help="Weight for cross-encoder term"),
    gamma: float | None = typer.Option(None, "--gamma", help="Weight for MMR diversity term"),
    url: str = typer.Option(
        DEFAULT_API,
        "--url",
        envvar=API_URL_ENV,
        show_default="env/localhost",
    ),
    retry: int = typer.Option(0, "--retry", help="Retry attempts on failure."),
) -> None:
    """Semantic search in the memory vector store."""

    async def _run_search() -> None:
        async with _client(url) as client:
            payload: dict[str, Any] = {
                "query": query,
                "top_k": top_k,
                "modality": modality,
            }
            if metadata_filter is not None:
                payload["metadata_filter"] = metadata_filter
            if level is not None:
                payload["level"] = level
            if alpha is not None:
                payload["alpha"] = alpha
            if beta is not None:
                payload["beta"] = beta
            if gamma is not None:
                payload["gamma"] = gamma

            rprint(f"[grey]POST {url}/memory/search …")
            resp = await _request_with_retry(
                client, "POST", "/memory/search", json=payload, retries=retry
            )
            resp.raise_for_status()
            results = resp.json()

            table = Table(title=f"Top-{top_k} results for '{query}'")
            table.add_column("Score", justify="right")
            table.add_column("Text", justify="left")

            for row in results:
                text_snip = row["text"][:80] + ("…" if len(row["text"]) > 80 else "")
                table.add_row(f"{row['score']:.2f}", text_snip)

            rprint(table)

    run(_run_search())


@app.command()
def consolidate(
    url: str = typer.Option(
        DEFAULT_API,
        "--url",
        envvar=API_URL_ENV,
        show_default="env/localhost",
    ),
    retry: int = typer.Option(0, "--retry", help="Retry attempts on failure."),
    idempotency_key: str | None = typer.Option(
        None,
        "--key",
        "--idempotency-key",
        help="Idempotency key to avoid duplicate submissions.",
    ),
) -> None:
    """Trigger memory consolidation on the server."""

    async def _run_consolidate() -> None:
        async with _client(url) as client:
            rprint(f"[grey]POST {url}/memory/consolidate …")
            headers = {"Idempotency-Key": idempotency_key or uuid.uuid4().hex}
            resp = await _request_with_retry(
                client,
                "POST",
                "/memory/consolidate",
                headers=headers,
                retries=retry,
            )
            resp.raise_for_status()
            data = resp.json()
            rprint(Panel(f"Consolidated {data.get('created', 0)} memories"))

    run(_run_consolidate())


@app.command()
def forget(
    url: str = typer.Option(
        DEFAULT_API,
        "--url",
        envvar=API_URL_ENV,
        show_default="env/localhost",
    ),
    retry: int = typer.Option(0, "--retry", help="Retry attempts on failure."),
    idempotency_key: str | None = typer.Option(
        None,
        "--key",
        "--idempotency-key",
        help="Idempotency key to avoid duplicate submissions.",
    ),
) -> None:
    """Forget low-value memories via the maintenance endpoint."""

    async def _run_forget() -> None:
        async with _client(url) as client:
            rprint(f"[grey]POST {url}/memory/forget …")
            headers = {"Idempotency-Key": idempotency_key or uuid.uuid4().hex}
            resp = await _request_with_retry(
                client,
                "POST",
                "/memory/forget",
                headers=headers,
                retries=retry,
            )
            resp.raise_for_status()
            data = resp.json()
            rprint(Panel(f"Deleted {data.get('deleted', 0)} memories", style="bold red"))

    run(_run_forget())


@app.command()
def delete(
    mem_id: str = typer.Argument(..., help="Memory ID to delete."),
    url: str = typer.Option(
        DEFAULT_API,
        "--url",
        envvar=API_URL_ENV,
        show_default="env/localhost",
    ),
    retry: int = typer.Option(0, "--retry", help="Retry attempts on failure."),
) -> None:
    """Delete a memory by ID."""

    async def _run_delete() -> None:
        async with _client(url) as client:
            rprint(f"[grey]DELETE {url}/memory/{mem_id} …")
            resp = await _request_with_retry(client, "DELETE", f"/memory/{mem_id}", retries=retry)
            resp.raise_for_status()
            rprint(Panel("Deleted ✔", style="bold red"))

    run(_run_delete())


@app.command()
def import_json(
    file: Path = typer.Argument(
        ..., exists=True, readable=True, help="JSONL file (one memory per line)."
    ),
    url: str = typer.Option(
        DEFAULT_API,
        "--url",
        envvar=API_URL_ENV,
        show_default="env/localhost",
    ),
    retry: int = typer.Option(0, "--retry", help="Retry attempts on failure."),
) -> None:
    """Bulk-import memories from a .jsonl file."""

    async def _run_import() -> None:
        async with _client(url) as client:
            semaphore = asyncio.Semaphore(8)

            async def _post(payload: dict[str, Any]) -> int:
                async with semaphore:
                    resp = await _request_with_retry(
                        client, "POST", "/memory/add", json=payload, retries=retry
                    )
                    resp.raise_for_status()
                    return 1

            added = 0
            tasks: list[asyncio.Task[int]] = []
            async with aopen(str(file), "r", encoding="utf-8") as fh:
                async for line in fh:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:  # pragma: no cover - simple log
                        rprint(f"[yellow]Skipping invalid JSON line: {exc}[/]")
                        continue

                    tasks.append(asyncio.create_task(_post(payload)))
                    if len(tasks) >= 8:
                        for t in asyncio.as_completed(tasks):
                            added += await t
                        tasks.clear()

            for t in asyncio.as_completed(tasks):
                added += await t

            rprint(Panel(f"Imported [bold green]{added}[/] memories"))

    run(_run_import())


if __name__ == "__main__":  # pragma: no cover
    app()  # Typer dispatch
