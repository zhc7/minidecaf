class TACVar:
    def __init__(self, name: str, value: int = 0, initialized: bool = False, size: int = 4) -> None:
        self.name = name
        self.value = value
        self.initialized = initialized
        self.size = size

    def printTo(self) -> None:
        print(f"var {self.name}", self.value if self.initialized else None)
