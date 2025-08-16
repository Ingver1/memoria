"""Admin routes exposed under `/api/v1/admin`."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, Request, status
from starlette.responses import Response

from memory_system.api.dependencies import (
    get_current_user,
    get_memory_store,
    get_token_manager,
    get_vector_store,
    oauth2_scheme,
)
from memory_system.api.middleware import MaintenanceModeMiddleware
from memory_system.core.store import audit_logger

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional fastapi security module
    from fastapi.security import SecurityScopes as _SecurityScopes
except Exception:  # pragma: no cover - stub fallback when fastapi not installed

    class _SecurityScopes:
        def __init__(self, scopes: list[str] | None = None) -> None:
            self.scopes = scopes or []


SecurityScopes = _SecurityScopes


async def _admin_user(
    token: str = Depends(oauth2_scheme),
    token_mgr: Any = Depends(get_token_manager),
) -> dict[str, Any]:
    """Dependency ensuring the caller has the ``admin`` scope."""
    scopes = SecurityScopes(["admin"])
    return await get_current_user(scopes, token, token_mgr)


router = APIRouter(prefix="/admin", tags=["Administration"])


def _maintenance(request: Request) -> MaintenanceModeMiddleware:
    """Retrieve the application's maintenance middleware instance."""
    if not hasattr(request.app.state, "maintenance"):
        request.app.state.maintenance = MaintenanceModeMiddleware(request.app)
    return cast("MaintenanceModeMiddleware", request.app.state.maintenance)


@router.get(
    "/maintenance-mode",
    summary="Get maintenance mode state",
    response_model=dict[str, bool],
)
async def maintenance_status(
    request: Request,
    _: dict[str, Any] = Depends(_admin_user),
) -> dict[str, bool]:
    """Check whether maintenance mode is currently enabled."""
    mw = _maintenance(request)
    return {"enabled": mw._enabled}


@router.post(
    "/maintenance-mode/enable",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Enable maintenance mode",
)
async def enable_maintenance(
    request: Request,
    _: dict[str, Any] = Depends(_admin_user),
) -> Response:
    """Switch maintenance mode **on** (returns 204 No Content on success)."""
    _maintenance(request).enable()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/maintenance-mode/disable",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Disable maintenance mode",
)
async def disable_maintenance(
    request: Request,
    _: dict[str, Any] = Depends(_admin_user),
) -> Response:
    """Switch maintenance mode **off** and restore normal operation."""
    _maintenance(request).disable()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user and associated data",
)
async def delete_user(user_id: str, request: Request) -> Response:
    """Remove all data linked to ``user_id`` and revoke associated keys."""
    store = get_memory_store(request)
    vector_store = get_vector_store(request)

    meta_store = cast("Any", getattr(store, "meta_store", store))
    ids: list[str] = []
    async for chunk in meta_store.search_iter(metadata_filters={"user_id": user_id}):
        ids.extend(m.id for m in chunk)

    for mid in ids:
        await meta_store.delete_memory(mid)

    if vector_store is not None and ids:
        delete_fn = getattr(vector_store, "delete", None)
        if inspect.iscoroutinefunction(delete_fn):
            await delete_fn(ids)
        elif callable(delete_fn):
            await asyncio.to_thread(delete_fn, ids)

    audit_logger.info("user.delete", extra={"user_id": user_id, "count": len(ids)})

    token_mgr = getattr(request.app.state, "token_manager", None)
    if token_mgr is not None:
        revoke_user = getattr(token_mgr, "revoke_user", None)
        if callable(revoke_user):
            revoke_user(user_id)
        else:
            revoke_token = getattr(token_mgr, "revoke_token", None)
            tokens_map = getattr(request.app.state, "user_tokens", {})
            if callable(revoke_token):
                for tok in tokens_map.get(user_id, []):
                    try:
                        revoke_token(tok)
                    except Exception as exc:
                        logger.warning("Token revoke failed: %s", exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
