"""Global test configuration for pytest."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Always load the bundled stub plugins
pytest_plugins = ("memoria._pytest_plugins",)
