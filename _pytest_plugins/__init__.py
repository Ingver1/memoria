"""Bundled pytest plugins."""

from importlib import import_module
import sys

pytest_plugins = (
    "pytest_asyncio",
    "pytest_cov",
)

# Register short plugin names for compatibility with ``pytest -p``
for _name in pytest_plugins:
    module = import_module(f"{__name__}.{_name}")
    sys.modules.setdefault(_name, module)
