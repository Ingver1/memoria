"""Authentication endpoints for issuing access tokens."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

try:  # pragma: no cover - optional fastapi security module
    from fastapi.security import OAuth2PasswordRequestForm as _OAuth2PasswordRequestForm
except Exception:  # pragma: no cover - stub fallback when fastapi not installed

    class _OAuth2PasswordRequestForm:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.username = ""
            self.password = ""
            self.scopes: list[str] = []


OAuth2PasswordRequestForm = _OAuth2PasswordRequestForm


from memory_system.api.dependencies import get_settings, get_token_manager
from memory_system.api.schemas import TokenResponse
from memory_system.settings import UnifiedSettings
from memory_system.utils.security import SecureTokenManager

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/token", summary="Issue access token")
async def issue_token(
    # FastAPI's ``Depends`` normally infers the dependency from the type
    # annotation when called without arguments.  Our lightweight test stubs
    # implement ``Depends`` as a simple function that *requires* the dependency
    # to be passed explicitly.  To keep the endpoint compatible with both the
    # real FastAPI implementation and the test environment, we provide the
    # dependency explicitly here.
    form: OAuth2PasswordRequestForm = Depends(OAuth2PasswordRequestForm),
    settings: UnifiedSettings = Depends(get_settings),
    token_mgr: SecureTokenManager = Depends(get_token_manager),
) -> TokenResponse:
    """Validate credentials and return an access token."""
    if form.password != settings.security.api_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password"
        )
    token = token_mgr.generate_token(form.username, scopes=form.scopes)
    return TokenResponse(access_token=token, token_type="bearer")
