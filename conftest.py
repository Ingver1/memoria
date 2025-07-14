"""Global test configuration for pytest."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Always load the bundled ``pytest_asyncio`` plugin for async test support
pytest_plugins = ("pytest_asyncio",)
