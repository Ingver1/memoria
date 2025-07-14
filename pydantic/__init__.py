from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any, Callable, TypeVar

__all__ = [
    "BaseModel",
    "BaseSettings",
    "ValidationError",
    "PositiveInt",
    "SecretStr",
    "Field",
    "field_validator",
    "ValidationInfo",
]

T = TypeVar("T")

class ValidationError(Exception):
    pass

class ValidationInfo:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or {}

class BaseModel:
    model_config: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        annotations = typing.get_type_hints(self.__class__)
        for name, ann in annotations.items():
            if name in data:
                value = data.pop(name)
            elif hasattr(self.__class__, name):
                value = getattr(self.__class__, name)
            else:
                continue
            if isinstance(value, dict) and isinstance(ann, type) and issubclass(ann, BaseModel):
                setattr(self, name, ann(**value))
            else:
                setattr(self, name, value)
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls: type[T], data: Any) -> T:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(**data)
        raise ValidationError("Invalid data")

    def model_dump(self, *, mode: str | None = None, exclude: set[str] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BaseModel):
                result[key] = value.model_dump()
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        if exclude:
            for field in exclude:
                result.pop(field, None)
        return result

    def model_dump_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), indent=indent)

class BaseSettings(BaseModel):
    pass

class SecretStr(str):
    def get_secret_value(self) -> str:
        return str(self)

PositiveInt = int

F = TypeVar("F", bound=Callable[..., Any])

def Field(default: Any = ..., **kwargs: Any) -> Any:
    if "default_factory" in kwargs:
        return kwargs["default_factory"]()
    return default

def field_validator(*_: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        return func
    return decorator
