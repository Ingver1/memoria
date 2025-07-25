import os
# Ensure pytest only loads the bundled stub plugins
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
