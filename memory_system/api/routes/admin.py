"""Admin routes exposed under `/api/v1/admin`."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request, status
from starlette.responses import Response

from memory_system.api.middleware import MaintenanceModeMiddleware

router = APIRouter(prefix="/admin", tags=["Administration"])


def _maintenance(request: Request) -> MaintenanceModeMiddleware:
    """Retrieve the application's maintenance middleware instance."""
    return cast(MaintenanceModeMiddleware, request.app.state.maintenance)


@router.get("/maintenance-mode", summary="Get maintenance mode state", response_model=dict)
async def maintenance_status(request: Request) -> dict[str, bool]:
    """Check whether maintenance mode is currently enabled."""
    mw = _maintenance(request)
    return {"enabled": mw._enabled}


@router.post(
    "/maintenance-mode/enable",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Enable maintenance mode",
)
async def enable_maintenance(request: Request) -> Response:
    """Switch maintenance mode **on** (returns 204 No Content on success)."""
    _maintenance(request).enable()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/maintenance-mode/disable",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Disable maintenance mode",
)
async def disable_maintenance(request: Request) -> Response:
    """Switch maintenance mode **off** and restore normal operation."""
    _maintenance(request).disable()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
