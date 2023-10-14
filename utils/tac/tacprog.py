from .tacfunc import TACFunc
from .tacvar import TACVar


# A TAC program consists of several TAC functions.
class TACProg:
    def __init__(self, funcs: list[TACFunc], variables: list[TACVar]) -> None:
        self.funcs = funcs
        self.variables = variables

    def printTo(self) -> None:
        for var in self.variables:
            var.printTo()
        for func in self.funcs:
            func.printTo()
