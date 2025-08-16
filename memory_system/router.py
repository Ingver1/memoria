from __future__ import annotations

import os
import re
import tomllib
import unicodedata as _ud
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, cast

# --- Configuration loading -------------------------------------------------


def _load_cfg() -> Mapping[str, Any]:
    """Load router config from ``pyproject.toml`` if available.

    Returns an immutable mapping to avoid accidental mutation.
    """
    try:
        cfg_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}

    tool = cast(Mapping[str, Any], data.get("tool", {}))
    memoria = cast(Mapping[str, Any], tool.get("memoria", {}))
    router = cast(Mapping[str, Any], memoria.get("router", {}))
    return router


_cfg: Final[Mapping[str, Any]] = _load_cfg()


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Accepts common truthy strings; falls back to ``default``.
    """
    val = os.getenv(name)
    if val is None:
        return bool(default)
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _cfg_int(key: str, default: int) -> int:
    raw = _cfg.get(key, default)
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def _cfg_str(key: str, default: str) -> str:
    raw = _cfg.get(key, default)
    if isinstance(raw, str):
        return raw
    return default


MIN_QUERY_LEN: Final[int] = int(
    os.getenv("MEMORIA_ROUTER_MIN_QUERY_LEN", str(_cfg_int("min_query_len", 6)))
)
NO_MEMORY_PREFIX: Final[str] = (
    os.getenv("MEMORIA_ROUTER_NO_MEMORY_PREFIX", _cfg_str("no_memory_prefix", "no memory:"))
    or "no memory:"
)
ENABLE_PERSONAL: Final[bool] = _env_bool(
    "MEMORIA_ROUTER_ENABLE_PERSONAL", bool(_cfg.get("enable_personal", True))
)
ENABLE_PROJECT: Final[bool] = _env_bool(
    "MEMORIA_ROUTER_ENABLE_PROJECT", bool(_cfg.get("enable_project", True))
)

# --- Trigger dictionaries --------------------------------------------------

PERSONAL_TRIGGERS = {
    "en": (
        "remember",
        "remind me",
        "my ",
        "for me",
    ),
    "ru": (
        "напомни",
        "мне ",
        "мой ",
        "моя ",
        "мои ",
        "моё ",
        "для меня",
        "сохрани мне",
    ),
    "es": (
        "recuerda",
        "recordarme",
        "recuérdame",
        "mi ",
    ),
}

PROJECT_TRIGGERS = {
    "en": (
        "repo",
        "repository",
        "pull request",
        "merge request",
        "issue",
        "ticket",
        "build",
        "deploy",
        "pipeline",
        "ci",
        "cd",
        "stacktrace",
        "traceback",
        "log",
        "error log",
    ),
    "ru": (
        "репозиторий",
        "репо",
        "пулл реквест",
        "мердж реквест",
        "таск",
        "тикет",
        "задача",
        "сборка",
        "деплой",
        "проект",
        "пайплайн",
        "ci",
        "cd",
        "стектрейс",
        "трейсбек",
        "лог",
        "логи",
        "лог ошибки",
    ),
    "es": (
        "repositorio",
        "pull request",
        "issue",
        "ticket",
        "compilación",
        "deploy",
        "proyecto",
        "pipeline",
        "ci",
        "cd",
        "traza",
        "registro",
        "log",
    ),
}

# --- Helpers ---------------------------------------------------------------


def _normalize(text: str) -> str:
    return _ud.normalize("NFKC", text).strip().lower()


def _compile_union_regex(phrases: Iterable[str]) -> re.Pattern[str]:
    """Compile a single regex that matches any of the given phrases.

    Uses simple substring matching (case-insensitive) for robustness across
    languages and tokens, as property tests may append arbitrary tails.
    """
    parts: list[str] = []
    for p in phrases:
        pp = p.strip()
        if not pp:
            continue
        parts.append(re.escape(pp))
    if not parts:
        return re.compile(r"(?!x)x", re.IGNORECASE)
    return re.compile("|".join(parts), re.IGNORECASE)


_PERSONAL_RX = _compile_union_regex(x for v in PERSONAL_TRIGGERS.values() for x in v)
_PROJECT_RX = _compile_union_regex(x for v in PROJECT_TRIGGERS.values() for x in v)


@dataclass(frozen=True)
class RouteDecision:
    """Decision about where to route and whether to retrieve.

    - ``target_channels``: ordered list of channels to enable.
    - ``use_retrieval``: whether retrieval should be performed at all.
    - ``is_global_query``: true for global/project queries (or when disabled).
    """

    target_channels: list[Channel]
    use_retrieval: bool
    is_global_query: bool

    # Backward-compatible view used by legacy callers/tests
    @property
    def targets(self) -> tuple[str, ...]:
        mapping = {Channel.PERSONAL: "personal", Channel.GLOBAL: "project"}
        return tuple(mapping[ch] for ch in self.target_channels)


class Channel(str, Enum):
    PERSONAL = "personal"
    PROJECT = "project"
    GLOBAL = "global"


# --- Router ---------------------------------------------------------------


class SimpleRouter:
    """
    Dependency-free router for two targets: 'personal' and 'project'.
    """

    # Public, introspected by tests
    MIN_QUERY_LEN: int = MIN_QUERY_LEN
    NO_MEMORY_PREFIX: str = NO_MEMORY_PREFIX
    PERSONAL_TRIGGERS: tuple[str, ...] = tuple(x for v in PERSONAL_TRIGGERS.values() for x in v)
    GLOBAL_TRIGGERS: tuple[str, ...] = tuple(x for v in PROJECT_TRIGGERS.values() for x in v)

    @staticmethod
    def decide(
        query: str,
        *,
        force_project_context: bool = False,
        enable_personal: bool = ENABLE_PERSONAL,
        enable_project: bool = ENABLE_PROJECT,
    ) -> RouteDecision:
        q = _normalize(query)

        # 1) opt-out
        if q.startswith(NO_MEMORY_PREFIX):
            return RouteDecision(target_channels=[], use_retrieval=False, is_global_query=True)

        # 2) too short
        if len(q) < MIN_QUERY_LEN:
            return RouteDecision(target_channels=[], use_retrieval=False, is_global_query=True)

        personal = False
        project = False

        # 3) triggers
        if enable_personal and _PERSONAL_RX.search(q):
            personal = True

        if enable_project and (_PROJECT_RX.search(q) or force_project_context):
            project = True

        # 4) decide final
        targets: list[Channel]
        if personal and project:
            targets = [Channel.PERSONAL, Channel.GLOBAL]
        elif personal:
            targets = [Channel.PERSONAL]
        elif project:
            targets = [Channel.GLOBAL]
        else:
            targets = []

        use_retrieval = len(targets) > 0
        # Consider query "global" unless it is personal-only.
        is_global_query = Channel.PERSONAL not in targets

        return RouteDecision(
            target_channels=targets, use_retrieval=use_retrieval, is_global_query=is_global_query
        )


__all__ = ["Channel", "RouteDecision", "SimpleRouter"]
