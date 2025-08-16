from __future__ import annotations

from typing import Any


class CliRunner:
    def invoke(self, *args: Any, **kwargs: Any):
        raise NotImplementedError
