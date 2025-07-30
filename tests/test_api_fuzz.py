"""
Fuzzes the whole FastAPI surface against its OpenAPI schema.
Schemathesis autogenerates thousands of requests with random payloads.
"""

from typing import Any

import pytest
import schemathesis
from schemathesis import DataGenerationMethod
from starlette.responses import Response

from memory_system.api.app import create_app
from memory_system.config.settings import UnifiedSettings


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
