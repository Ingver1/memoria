from typing import Callable
class DataGenerationMethod:
    fuzzed: str = 'fuzzed'

class _Case:
    def call_asgi(self, app: object) -> object:
        from starlette.responses import Response
        return Response()

    def validate_response(self, response: object) -> None:
        pass

class _Schema:
    from typing import Callable
    def parametrize(self) -> object:
        def decorator(func: Callable[[_Case], None]) -> object:
            def wrapper() -> None:
                case = _Case()
                func(case)
            return wrapper
        return decorator

def from_path(path: str, data_generation_methods: object = None) -> object:
    return _Schema()

__all__ = ['from_path', 'DataGenerationMethod']
