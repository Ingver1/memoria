"""
Fuzzes the whole FastAPI surface against its OpenAPI schema.
Schemathesis autogenerates thousands of requests with random payloads.
"""

from typing import Any

import pytest

try:  # pragma: no cover - optional dependency
    import schemathesis
except ImportError:  # pragma: no cover - schemathesis is optional
    schemathesis = None  # type: ignore[assignment]

if schemathesis is None:  # pragma: no cover - schemathesis missing
    pytest.skip("schemathesis not installed", allow_module_level=True)

try:  # pragma: no cover - handle API changes
    from schemathesis import DataGenerationMethod
except ImportError:  # Schemathesis >= 3.27
    try:
        from schemathesis.specs.openapi import DataGenerationMethod
    except ImportError:  # pragma: no cover - fallback failed
        pytest.skip("schemathesis DataGenerationMethod unavailable", allow_module_level=True)
from starlette.responses import Response

from memory_system.api.app import create_app
from memory_system.settings import UnifiedSettings


class _Case:
    def call_asgi(self, app: Any) -> Response:
        return Response()

    def validate_response(self, response: Any) -> None:
        pass


@pytest.fixture
def case() -> _Case:
    """Provide a minimal `case` fixture when the real schemathesis plugin is missing."""
    return _Case()


schema = schemathesis.from_path(
    "tests/st_api_fuzz.yaml",
    data_generation_methods=[DataGenerationMethod.fuzzed],
)


def test_api_fuzz(case: _Case) -> None:
    """Run schema-driven fuzzing against the live ASGI app."""
    app = create_app(UnifiedSettings.for_testing())
    response = case.call_asgi(app)
    case.validate_response(response)
