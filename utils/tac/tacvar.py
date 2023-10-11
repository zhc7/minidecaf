from typing import Union, List


class TACVar:
    def __init__(self, name: str, value: Union[int, List[int]] = 0, initialized: bool = False, size: int = 4) -> None:
        self.name = name
        self._value = value
        self.initialized = initialized
        self.size = size

    @property
    def value(self):
        return self._value if isinstance(self._value, int) else ", ".join([str(val) for val in self._value])

    def printTo(self) -> None:
        print(f"var {self.name}", self.value if self.initialized else None)
