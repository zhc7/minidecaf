from typing import Protocol, TypeVar, cast

from frontend.ast.node import Node, NullType
from frontend.ast.tree import *
from frontend.ast.visitor import RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract 
syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[Scope, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # Global scope. You don't have to consider it until Step 6.
        program.globalScope = GlobalScope
        ctx = program.globalScope  # Scope(program.globalScope, program.globalScope)

        program.accept(self, ctx)
        return program

    def visitProgram(self, program: Program, ctx: Scope) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError

        for func in program.functions().values():
            func.accept(self, ctx)

    def visitParameter(self, param: Parameter, ctx: Scope) -> None:
        if ctx.lookup(param.ident.value, True) is None:
            param_symbol = VarSymbol(param.ident.value, param.var_t.type)
            ctx.declare(param_symbol)
            param.symbol = param_symbol
        else:
            raise DecafDeclConflictError(param.ident.value)

    def visitFunction(self, func: Function, ctx: Scope) -> None:
        assert ctx.isGlobalScope()
        assert ctx.lookup(func.ident.value, True) is None
        func_symbol = FuncSymbol(func.ident.value, func.ret_t.type, ctx)
        for param in func.params:
            func_symbol.addParaType(param.var_t.type)
        ctx.declare(func_symbol)
        func.symbol = func_symbol
        ctx = Scope(ScopeKind.LOCAL, ctx)
        for param in func.params:
            param.accept(self, ctx)
        if func.body is None:
            return
        self.visitBlock(func.body, ctx, False)

    def visitBlock(self, block: Block, ctx: Scope, new_scope: bool = True) -> None:
        if new_scope:
            ctx = Scope(ScopeKind.LOCAL, ctx)
        for child in block:
            child.accept(self, ctx)

    def visitReturn(self, stmt: Return, ctx: Scope) -> None:
        stmt.expr.accept(self, ctx)

    """
    def visitFor(self, stmt: For, ctx: Scope) -> None:

    1. Open a local scope for stmt.init.
    2. Visit stmt.init, stmt.cond, stmt.update.
    3. Open a loop in ctx (for validity checking of break/continue)
    4. Visit body of the loop.
    5. Close the loop and the local scope.
    """

    def visitIf(self, stmt: If, ctx: Scope) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: Scope) -> None:
        stmt.cond.accept(self, ctx)
        ctx = Scope(ScopeKind.LOCAL, ctx, True)
        stmt.body.accept(self, ctx)

    def visitFor(self, stmt: For, ctx: Scope) -> None:
        ctx = Scope(ScopeKind.LOCAL, ctx, True)
        stmt.init.accept(self, ctx)
        stmt.cond.accept(self, ctx)
        stmt.after.accept(self, ctx)
        stmt.body.accept(self, ctx)

    def visitBreak(self, stmt: Break, ctx: Scope) -> None:
        """
        You need to check if it is currently within the loop.
        To do this, you may need to check 'visitWhile'.

        if not in a loop:
            raise DecafBreakOutsideLoopError()
        """
        if not ctx.in_loop:
            raise DecafBreakOutsideLoopError()

    """
    def visitContinue(self, stmt: Continue, ctx: Scope) -> None:
    
    1. Refer to the implementation of visitBreak.
    """

    def visitContinue(self, stmt: Continue, ctx: Scope) -> None:
        if not ctx.in_loop:
            raise DecafContinueOutsideLoopError()

    def visitDeclaration(self, decl: Declaration, ctx: Scope) -> None:
        """
        1. Use ctx.lookup to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        if ctx.lookup(decl.ident.value, True) is None:
            var_symbol = VarSymbol(decl.ident.value, decl.var_t.type)
            ctx.declare(var_symbol)
            decl.symbol = var_symbol
            if decl.init_expr is not NULL:
                decl.init_expr.accept(self, ctx)
        else:
            raise DecafDeclConflictError(decl.ident.value)

    def visitAssignment(self, expr: Assignment, ctx: Scope) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCall(self, call: Call, ctx: Scope) -> None:
        symbol = ctx.lookup(call.ident.value)
        if symbol is None or not symbol.isFunc:
            raise DecafUndefinedFuncError(call.ident.value)
        assert isinstance(symbol, FuncSymbol)
        if symbol.parameterNum != len(call.args):
            raise DecafUndefinedFuncError(call.ident.value)
        for arg in call.args:
            arg.accept(self, ctx)

    def visitUnary(self, expr: Unary, ctx: Scope) -> None:
        expr.operand.accept(self, ctx)

    def visitBinary(self, expr: Binary, ctx: Scope) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCondExpr(self, expr: ConditionExpression, ctx: Scope) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)

    def visitIdentifier(self, ident: Identifier, ctx: Scope) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        symbol = ctx.lookup(ident.value)
        if symbol is None:
            raise DecafUndefinedVarError(ident.value)
        ident.symbol = symbol

    def visitIntLiteral(self, expr: IntLiteral, ctx: Scope) -> None:
        value = expr.value
        if value > MAX_INT:
            raise DecafBadIntValueError(value)
