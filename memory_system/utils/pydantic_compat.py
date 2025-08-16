# Minimal subset of Pydantic used for tests
from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar, cast


def import_pydantic() -> Any:
    """Return the real :mod:`pydantic` if available, else a small shim."""
    try:
        import pydantic as _pyd
    except ImportError:  # pragma: no cover
        _pyd = sys.modules[__name__]
    return _pyd


__all__ = [
    "AliasChoices",
    "BaseModel",
    "ConfigDict",
    "Field",
    "NonNegativeInt",
    "PositiveInt",
    "SecretStr",
    "ValidationError",
    "ValidationInfo",
    "field_validator",
    "import_pydantic",
    "model_validator",
]

MISSING = object()


class ValidationError(ValueError):
    """Exception raised for validation errors."""


class AliasChoices:
    def __init__(self, *choices: str) -> None:
        self.choices = list(choices)


class ValidationInfo:
    def __init__(self, *, data: dict[str, Any]) -> None:
        self.data = data


class SecretStr:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "SecretStr('********')"


class FieldInfo:
    def __init__(
        self,
        default: Any = MISSING,
        *,
        default_factory: Callable[[], Any] | None = None,
        validation_alias: AliasChoices | str | None = None,
        serialization_alias: str | None = None,
        **_: Any,
    ) -> None:
        self.default = default
        self.default_factory = default_factory
        if validation_alias is None:
            self.aliases: list[str] = []
        elif isinstance(validation_alias, AliasChoices):
            self.aliases = list(validation_alias.choices)
        else:
            self.aliases = [validation_alias]
        self.serialization_alias = serialization_alias


def Field(
    default: Any = MISSING, *, default_factory: Callable[[], Any] | None = None, **kwargs: Any
) -> FieldInfo:
    if default is ...:  # treat ellipsis as "required" marker
        default = MISSING
    return FieldInfo(default, default_factory=default_factory, **kwargs)


class _PydanticCompatAttrs(Protocol):
    """Protocol for classes with temporary Pydantic shim attributes."""

    __pydantic_fields__: dict[str, tuple[Any, FieldInfo]]
    __pydantic_validators__: dict[str, list[tuple[Callable[..., Any], bool]]]
    __pydantic_model_validators__: list[tuple[Callable[..., Any], bool]]


def ConfigDict(**kwargs: Any) -> dict[str, Any]:
    """
    Simplified stand-in for :func:`pydantic.ConfigDict`.

    Returns a plain dictionary with the provided keyword arguments.  This
    mirrors the behaviour needed by the project's tests without depending on
    the real Pydantic implementation.
    """
    return dict(**kwargs)


class NonNegativeInt(int):
    pass


class PositiveInt(int):
    pass


def field_validator(*fields: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        raw: Any = func
        is_cls = isinstance(func, classmethod)
        if is_cls:
            raw = cast("Any", func).__func__
        raw.__field_validators__ = fields
        raw.__validator_is_classmethod__ = is_cls
        return func

    return decorator


def model_validator(*, mode: str = "after") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register a model-level validator.

    Only ``mode="after"`` is recognised by this lightweight shim which runs
    validators after field parsing.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        raw: Any = func
        is_cls = isinstance(func, classmethod)
        if is_cls:
            raw = cast("Any", func).__func__
        raw.__model_validator__ = mode
        raw.__validator_is_classmethod__ = is_cls
        return func

    return decorator


T = TypeVar("T", bound="BaseModel")


class BaseModel:
    model_config: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        cls = self.__class__
        cls_fields = _get_class_fields(cls)
        validators = _get_class_validators(cls)
        pyd_cls = cast("_PydanticCompatAttrs", cast("Any", cls))
        model_validators = cast(
            "list[tuple[Callable[..., Any], bool]]",
            getattr(pyd_cls, "__pydantic_model_validators__", []),
        )
        values: dict[str, Any] = {}
        for name, (typ, finfo) in cls_fields.items():
            if name in data:
                val = data.pop(name)
            else:
                for alias in finfo.aliases:
                    if alias in data:
                        val = data.pop(alias)
                        break
                else:
                    if finfo.default is not MISSING:
                        val = finfo.default
                    elif finfo.default_factory is not None:
                        val = finfo.default_factory()
                    else:
                        raise ValidationError(f"{name} is required")
            val = _basic_validate(name, typ, val)
            values[name] = val
        for extra_key in list(data.keys()):
            values[extra_key] = data[extra_key]
        for name, val in values.items():
            super().__setattr__(name, val)
        for name, val in list(values.items()):
            for func, is_cls in validators.get(name, []):
                try:
                    sig = inspect.signature(func)
                    if is_cls:
                        if len(sig.parameters) == 2:
                            val = func(cls, val)
                        else:
                            val = func(cls, val, ValidationInfo(data=values))
                    elif len(sig.parameters) == 1:
                        val = func(val)
                    else:
                        val = func(val, ValidationInfo(data=values))
                except ValueError as exc:  # pragma: no cover - simplicity
                    raise ValidationError(str(exc)) from exc
                values[name] = val
                super().__setattr__(name, val)

        for func, is_cls in model_validators:
            try:
                sig = inspect.signature(func)
                if is_cls:
                    if len(sig.parameters) == 2:
                        res = func(cls, self)
                    else:
                        res = func(cls, self, ValidationInfo(data=values))
                elif len(sig.parameters) == 1:
                    res = func(self)
                else:
                    res = func(self, ValidationInfo(data=values))
            except ValueError as exc:  # pragma: no cover - simplicity
                raise ValidationError(str(exc)) from exc
            if isinstance(res, BaseModel) and res is not self:
                for k, v in res.__dict__.items():
                    values[k] = v
                    super().__setattr__(k, v)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - immutability
        cfg = getattr(self, "model_config", {})
        if cfg.get("frozen") and name in self.__dict__:
            raise ValueError(f"{self.__class__.__name__} is immutable")
        super().__setattr__(name, value)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        cls = self.__class__
        cls_fields = _get_class_fields(cls)
        result: dict[str, Any] = {}
        for name, (_, finfo) in cls_fields.items():
            val = getattr(self, name)
            if isinstance(val, BaseModel):
                val = val.model_dump(mode=mode)
            elif isinstance(val, SecretStr):
                val = val.get_secret_value()
            elif isinstance(val, datetime) and mode == "json":
                val = val.isoformat()
            if isinstance(val, Path):
                val = str(val)
            key = finfo.serialization_alias or name
            result[key] = val
        return result

    def model_copy(self, update: dict[str, Any] | None = None) -> BaseModel:
        data = self.model_dump()
        if update:
            data.update(update)
        return self.__class__(**data)

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


def _get_class_fields(cls: type) -> dict[str, tuple[Any, FieldInfo]]:
    if "__pydantic_fields__" in vars(cls):
        return cast("_PydanticCompatAttrs", cast("Any", cls)).__pydantic_fields__
    annotations = vars(cls).get("__annotations__", {})
    fields_dict: dict[str, tuple[Any, FieldInfo]] = {}
    module_globals = vars(sys.modules.get(cls.__module__, object()))
    for name, typ in annotations.items():
        if isinstance(typ, str):  # handle forward references from future annotations
            typ = eval(typ, module_globals)
        default = getattr(cls, name, MISSING)
        finfo = default if isinstance(default, FieldInfo) else FieldInfo(default)
        fields_dict[name] = (typ, finfo)
    cast("Any", cls).__pydantic_fields__ = fields_dict
    return fields_dict


def _get_class_validators(cls: type) -> dict[str, list[tuple[Callable[..., Any], bool]]]:
    if "__pydantic_validators__" in vars(cls) and "__pydantic_model_validators__" in vars(cls):
        return cast("_PydanticCompatAttrs", cast("Any", cls)).__pydantic_validators__
    validators: dict[str, list[tuple[Callable[..., Any], bool]]] = {}
    model_validators: list[tuple[Callable[..., Any], bool]] = []
    for attr in cls.__dict__.values():
        func = attr
        is_cls = False
        if isinstance(attr, classmethod):
            func = attr.__func__
            is_cls = True
        fields = getattr(func, "__field_validators__", None)
        if fields:
            for field in fields:
                validators.setdefault(field, []).append((func, is_cls))
        if getattr(func, "__model_validator__", None):
            model_validators.append((func, is_cls))
    cast("Any", cls).__pydantic_validators__ = validators
    cast("Any", cls).__pydantic_model_validators__ = model_validators
    return validators


def _basic_validate(name: str, typ: Any, value: Any) -> Any:
    if typ is PositiveInt:
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{name} must be > 0")
    elif typ is NonNegativeInt:
        if not isinstance(value, int) or value < 0:
            raise ValidationError(f"{name} must be >= 0")
    elif typ is SecretStr and not isinstance(value, SecretStr):
        return SecretStr(str(value))
    elif isinstance(typ, type) and issubclass(typ, BaseModel) and isinstance(value, dict):
        # Support nested ``BaseModel`` types by recursively validating dict values.
        # This mirrors Pydantic's behaviour where sub-model fields are parsed into
        # their respective model classes rather than leaving them as plain dicts.
        return typ.model_validate(value)
    return value
