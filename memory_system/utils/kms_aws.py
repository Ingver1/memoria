from .security import KeyManagementBackend, ManagedKey

class AWSKMSBackend(KeyManagementBackend):
    def __init__(self, key_id: str, region: str) -> None:
        pass

    def load_all(self) -> list[ManagedKey]:
        return []

    def save(self, key: ManagedKey) -> None:
        pass

    def delete(self, key_id: str) -> None:
        pass
