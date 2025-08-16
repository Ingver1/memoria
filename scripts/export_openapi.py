#!/usr/bin/env python3
"""Generate OpenAPI schema for the Unified Memory System API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory_system.api.app import app


def main() -> None:
    schema: dict[str, Any] = app.openapi()
    output = Path("openapi.json")
    output.write_text(json.dumps(schema, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
