from memory_system.utils.pydantic_compat import import_pydantic

_pyd = import_pydantic()
AliasChoices = _pyd.AliasChoices
BaseModel = _pyd.BaseModel
ConfigDict = _pyd.ConfigDict
Field = _pyd.Field
model_validator = _pyd.model_validator
NonNegativeInt = _pyd.NonNegativeInt
PositiveInt = _pyd.PositiveInt
ValidationError = _pyd.ValidationError

__all__ = [
    "AliasChoices",
    "BaseModel",
    "ConfigDict",
    "Field",
    "NonNegativeInt",
    "PositiveInt",
    "ValidationError",
    "model_validator",
]
