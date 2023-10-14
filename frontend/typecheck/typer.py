from frontend.ast.tree import *
from frontend.ast.visitor import Visitor
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.type.array import ArrayType
from utils.error import *

"""
The typer phase: type check abstract syntax tree.
"""


class Typer(Visitor[None, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        for decl in program.declarations():
            decl.accept(self, None)
        for func in program.functions().values():
            func.accept(self, None)
        return program

    def visitFunction(self, that: Function, ctx: T) -> None:
        that.body.accept(self, ctx)
        if that.ret_t.type != that.body.type:
            raise DecafTypeMismatchError()

    def visitBlock(self, block: Block, ctx: T) -> None:
        block.type = INT
        for child in block.children:
            child.accept(self, ctx)
            if isinstance(child, Return):
                block.type = child.type

    def visitIf(self, that: If, ctx: T) -> None:
        that.cond.accept(self, ctx)
        that.then.accept(self, ctx)
        if that.otherwise != NULL:
            that.otherwise.accept(self, ctx)

    def visitWhile(self, that: While, ctx: T) -> None:
        that.cond.accept(self, ctx)
        that.body.accept(self, ctx)

    def visitFor(self, that: For, ctx: T) -> None:
        that.init.accept(self, ctx)
        that.cond.accept(self, ctx)
        that.after.accept(self, ctx)
        that.body.accept(self, ctx)

    def visitAssignment(self, assignment: Assignment, ctx: T) -> None:
        assignment.rhs.accept(self, ctx)
        assignment.lhs.accept(self, ctx)
        if assignment.lhs.type != assignment.rhs.type:
            raise DecafTypeMismatchError()
        assignment.type = assignment.lhs.type

    def visitDeclaration(self, that: Declaration, ctx: T) -> None:
        if that.init_expr != NULL:
            that.init_expr.accept(self, ctx)
            if isinstance(that.var_t.type, ArrayType):
                if not isinstance(that.var_t.type, ArrayType):
                    raise DecafTypeMismatchError()
                if not isinstance(that.init_expr, ArrayInit):
                    raise DecafTypeMismatchError()
                if that.init_expr.type is not None:
                    if that.init_expr.type != that.var_t.type.full_indexed:
                        raise DecafTypeMismatchError()
            elif that.init_expr.type != that.var_t.type:
                raise DecafTypeMismatchError()

    def visitArrayInit(self, that: ArrayInit, ctx: T) -> None:
        type_ = None
        for expr in that.children:
            expr.accept(self, ctx)
            if type_ is None:
                type_ = expr.type
            elif type_ != expr.type:
                raise DecafTypeMismatchError()
        that.type = type_

    def visitIntLiteral(self, that: IntLiteral, ctx: T) -> None:
        that.type = INT

    def visitUnary(self, that: Unary, ctx: T) -> None:
        that.operand.accept(self, ctx)
        that.type = that.operand.type

    def visitBinary(self, that: Binary, ctx: T) -> None:
        that.lhs.accept(self, ctx)
        that.rhs.accept(self, ctx)
        if that.lhs.type != that.rhs.type:
            raise DecafTypeMismatchError()
        if that.lhs.type not in [INT]:
            raise DecafTypeMismatchError()
        that.type = that.lhs.type

    def visitCondExpr(self, that: ConditionExpression, ctx: T) -> Optional[U]:
        that.cond.accept(self, ctx)
        that.then.accept(self, ctx)
        that.otherwise.accept(self, ctx)
        if that.then.type != that.otherwise.type:
            raise DecafTypeMismatchError()
        that.type = that.then.type

    def visitCall(self, that: Call, ctx: T) -> None:
        assert isinstance(that.ident.symbol, FuncSymbol)
        types = that.ident.symbol.para_type
        for t, arg in zip(types, that.args):
            arg.accept(self, ctx)
            if arg.type != t:
                raise DecafTypeMismatchError()
        that.type = that.ident.symbol.type

    def visitIdentifier(self, that: Identifier, ctx: T) -> None:
        that.type = that.symbol.type

    def visitArrayIndex(self, that: ArrayIndex, ctx: T):
        that.base.accept(self, ctx)
        that.index.accept(self, ctx)
        if that.index.type != INT:
            raise DecafTypeMismatchError()
        if not isinstance(that.base.type, ArrayType):
            raise DecafTypeMismatchError()
        that.type = that.symbol.type

    def visitReturn(self, that: Return, ctx: T) -> None:
        that.expr.accept(self, ctx)
        that.type = that.expr.type
