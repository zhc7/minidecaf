from enum import Enum, auto, unique
from typing import Optional

from frontend.symbol.symbol import Symbol

"""
A scope stores the mapping from names to symbols. There are two kinds of scopes:
    global scope: stores global variables and functions
    local scope: stores local variables of a code block
"""


@unique
class ScopeKind(Enum):
    GLOBAL = auto()
    LOCAL = auto()


class Scope:
    def __init__(self, kind: ScopeKind, father: Optional["Scope"], in_loop: bool = False) -> None:
        self.kind = kind
        self.symbols = {}
        self.father = father
        self.in_loop = (father and father.in_loop) or in_loop

    # To check if a symbol is declared in the scope.
    def containsKey(self, key: str) -> bool:
        return key in self.symbols or (self.father is not None and self.father.containsKey(key))

    # To get a symbol via its name.
    def get(self, key: str) -> Symbol:
        return self.symbols.get(key, None) or self.father.get(key)

    # To declare a symbol.
    def declare(self, symbol: Symbol) -> None:
        self.symbols[symbol.name] = symbol
        symbol.setDomain(self)

    # To check if this is a global scope.
    def isGlobalScope(self) -> bool:
        return False

    # To get a symbol if declared in the scope
    def lookup(self, name: str, strict: bool = False) -> Optional[Symbol]:
        if strict:
            if name in self.symbols:
                return self.symbols[name]
            return None
        if self.containsKey(name):
            return self.get(name)
        return None
