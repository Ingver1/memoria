import pytest
from fastapi import HTTPException

from memory_system.api.utils import validate_text_length


def test_validate_text_length_raises_http_exception() -> None:
    with pytest.raises(HTTPException):
        validate_text_length("a" * 11, 10)
