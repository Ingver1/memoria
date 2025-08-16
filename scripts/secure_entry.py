from __future__ import annotations

import os

import uvicorn

from memory_system.utils.security import CryptoContext


def main() -> None:
    kms_key_id = os.getenv("AI_SECURITY__KMS_KEY_ID")
    if kms_key_id:
        os.environ.setdefault("AWS_KMS_KEY_ID", kms_key_id)
    os.environ.setdefault("AI_SECURITY__ENCRYPT_AT_REST", "true")
    ctx = CryptoContext.from_env()
    os.environ.setdefault("AI_SECURITY__ENCRYPTION_KEY", ctx.get_active_key())
    uvicorn.run(
        "memory_system.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()
