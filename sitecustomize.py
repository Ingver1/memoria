import os
import sys
from pathlib import Path

# Add the repository root to ``sys.path`` so bundled plugins can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-import the bundled ``pytest_asyncio`` plugin so ``pytest -p pytest_asyncio``
# works even when running the ``pytest`` entry point directly.
try:  # pragma: no cover - only executed during test discovery
    import pytest_asyncio  # noqa: F401
except Exception:
    pass

# Ensure pytest only loads the bundled stub plugins
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
