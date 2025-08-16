# Development Guide

## Testing prerequisites

- Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to prevent auto-loading of unrelated pytest plugins.
- Install `pytest-asyncio` to enable asynchronous test fixtures.
- Some tests rely on optional packages such as `numpy` and `httpx`; install these extras to run the full suite.

