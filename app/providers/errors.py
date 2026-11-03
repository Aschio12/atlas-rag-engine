class ProviderError(Exception):
    """A provider call failed. Messages never contain credentials."""

    def __init__(self, provider: str, message: str, status: int | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status
