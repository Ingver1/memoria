class _Hasher:
    def __init__(self, data: bytes = b"") -> None:
        self.data = data

    def update(self, more: bytes) -> None:
        self.data += more

    def digest(self, length: int = 32) -> bytes:
        return b"\x00" * length


def blake3(data: bytes = b"") -> _Hasher:
    return _Hasher(data)
