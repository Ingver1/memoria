import importlib.util
import os


def test_pytest_plugin_autoload_disabled(pytestconfig):
    """Ensure pytest does not auto-load unrelated third-party plugins."""
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    loaded = {name for name, _ in pytestconfig.pluginmanager.list_name_plugin()}
    for plugin in ("pytest_cov", "pytest_mock", "pytest_xdist"):
        if importlib.util.find_spec(plugin) is not None:
            assert not any(name.startswith(plugin) for name in loaded)
