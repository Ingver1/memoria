from __future__ import annotations

from typing import Any, Mapping


class Response:
    def __init__(
        self,
        content: Any = None,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = dict(headers) if headers else {}
        self.media_type = media_type

    def json(self) -> Any:
        return self.content

class JSONResponse(Response):
    pass
