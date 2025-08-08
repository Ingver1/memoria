import subprocess
import sys


def test_docstrings() -> None:
    """Ensure memory_system.core has docstrings via ruff's D1 checks."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "memory_system/core", "--select", "D1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
