class DataGenerationMethod:
    fuzzed = 'fuzzed'

class _Case:
    def call_asgi(self, app):
        from starlette.responses import Response
        return Response()

    def validate_response(self, response):
        pass

class _Schema:
    def parametrize(self):
        def decorator(func):
            def wrapper():
                case = _Case()
                func(case)
            return wrapper
        return decorator

def from_path(path: str, data_generation_methods=None):
    return _Schema()

__all__ = ['from_path', 'DataGenerationMethod']
