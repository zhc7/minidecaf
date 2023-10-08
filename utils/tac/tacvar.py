class TACVar:
    def __init__(self, name: str, value: int = 0, initialized: bool = False) -> None:
        self.name = name
        self.value = value
        self.initialized = initialized

    def printTo(self) -> None:
        print(f"var {self.name}", self.value if self.initialized else None)
