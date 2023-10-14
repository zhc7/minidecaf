from typing import Union, Optional, List

from frontend.ast import node
from frontend.ast import tree
from frontend.ast.node import NULL
from frontend.ast.tree import (
    Program,
    IntLiteral,
    ArrayInit,
    ConditionExpression,
    For,
    If,
    Assignment,
    ArrayDeclaration,
    ArrayIndex,
    While,
    Break,
    Continue,
    Identifier,
    Block,
    Declaration,
)
from frontend.ast.visitor import Visitor
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from utils.label.blocklabel import BlockLabel
from utils.label.funclabel import FuncLabel
from utils.label.label import Label
from utils.tac import tacinstr
from utils.tac import tacop
from utils.tac.tacfunc import TACFunc
from utils.tac.tacinstr import Mark, Memo, LoadSymbol, Load, Alloc, Memset, TACInstr
from utils.tac.tacop import TacUnaryOp, TacBinaryOp, CondBranchOp
from utils.tac.tacprog import TACProg
from utils.tac.tacvar import TACVar
from utils.tac.tacvisitor import TACVisitor
from utils.tac.temp import Temp

"""
The TAC generation phase: translate the abstract syntax tree into three-address code.
"""


class LabelManager:
    """
    A global label manager (just a counter).
    We use this to create unique (block) labels across functions.
    """

    def __init__(self):
        self.nextTempLabelId = 0

    def freshLabel(self) -> BlockLabel:
        self.nextTempLabelId += 1
        return BlockLabel(str(self.nextTempLabelId))


class TACFuncEmitter(TACVisitor):
    """
    Translates a minidecaf (AST) function into low-level TAC function.
    """

    def __init__(
        self, entry: FuncLabel, numArgs: int, labelManager: LabelManager
    ) -> None:
        self.labelManager = labelManager
        self.func = TACFunc(entry, numArgs)
        self.visitLabel(entry)
        self.nextTempId = 0

        self.continueLabelStack = []
        self.breakLabelStack = []

        # load params
        self.func.add(tacinstr.LoadParams([Temp(i) for i in range(numArgs)]))

        # mark left or right value
        self.right = True

    # To get a fresh new temporary variable.
    def freshTemp(self) -> Temp:
        temp = Temp(self.nextTempId)
        self.nextTempId += 1
        return temp

    # To get a fresh new label (for jumping and branching, etc).
    def freshLabel(self) -> Label:
        return self.labelManager.freshLabel()

    # To count how many temporary variables have been used.
    def getUsedTemp(self) -> int:
        return self.nextTempId

    # In fact, the following methods can be named 'appendXXX' rather than 'visitXXX'.
    # E.g., by calling 'visitAssignment', you add an assignment instruction at the end of current function.
    def visitAssignment(self, dst: Temp, src: Temp) -> Temp:
        self.func.add(tacinstr.Assign(dst, src))
        return src

    def visitAddrAssignment(self, addr: Temp, src: Temp, offset: int = 0) -> Temp:
        self.func.add(tacinstr.AddrAssign(addr, src, offset))
        return src

    def visitLoadImm(self, value: Union[int, str]) -> Temp:
        temp = self.freshTemp()
        self.func.add(tacinstr.LoadImm4(temp, value))
        return temp

    def emitUnary(self, op: TacUnaryOp, operand: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(tacinstr.Unary(op, temp, operand))
        return temp

    def visitUnarySelf(self, op: TacUnaryOp, operand: Temp) -> None:
        self.func.add(tacinstr.Unary(op, operand, operand))

    def emitBinary(self, op: TacBinaryOp, lhs: Temp, rhs: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(tacinstr.Binary(op, temp, lhs, rhs))
        return temp

    def visitBinarySelf(self, op: TacBinaryOp, lhs: Temp, rhs: Temp) -> None:
        self.func.add(tacinstr.Binary(op, lhs, lhs, rhs))

    def visitBranch(self, target: Label) -> None:
        self.func.add(tacinstr.Branch(target))

    def emitCondBranch(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        self.func.add(tacinstr.CondBranch(op, cond, target))

    def visitReturn(self, value: Optional[Temp]) -> None:
        self.func.add(tacinstr.Return(value))

    def visitLabel(self, label: Label) -> None:
        self.func.add(Mark(label))

    def visitMemo(self, content: str) -> None:
        self.func.add(Memo(content))

    def visitGlobalVar(self, symbol: VarSymbol) -> Temp:
        addr = self.freshTemp()
        self.func.add(LoadSymbol(addr, symbol.name))
        return addr

    def visitAddr(self, addr: Temp) -> Temp:
        dst = self.freshTemp()
        self.func.add(Load(dst, addr))
        return dst

    def visitAlloc(self, size: int) -> Temp:
        temp = self.freshTemp()
        self.func.add(Alloc(temp, size))
        return temp

    def emitCall(self, func: FuncLabel, args: List[Temp]) -> Temp:
        temp = self.freshTemp()
        self.func.add(tacinstr.Call(func, args, temp))
        return temp

    def visitArrayInit(self, addr: Temp, size: int, init: List[Temp]):
        temp = self.visitLoadImm(int(size / 4))
        self.func.add(Memset(addr, temp))
        for i, tmp in enumerate(init):
            self.visitAddrAssignment(addr, tmp, 4 * i)

    def visitRaw(self, instr: TACInstr) -> None:
        self.func.add(instr)

    def visitEnd(self) -> TACFunc:
        if (len(self.func.instrSeq) == 0) or (not self.func.instrSeq[-1].isReturn()):
            self.func.add(tacinstr.Return(None))
        self.func.tempUsed = self.getUsedTemp()
        return self.func

    # To open a new loop (for break/continue statements)
    def openLoop(self, breakLabel: Label, continueLabel: Label) -> None:
        self.breakLabelStack.append(breakLabel)
        self.continueLabelStack.append(continueLabel)

    # To close the current loop.
    def closeLoop(self) -> None:
        self.breakLabelStack.pop()
        self.continueLabelStack.pop()

    # To get the label for 'break' in the current loop.
    def getBreakLabel(self) -> Label:
        return self.breakLabelStack[-1]

    # To get the label for 'continue' in the current loop.
    def getContinueLabel(self) -> Label:
        return self.continueLabelStack[-1]


class TACGen(Visitor[TACFuncEmitter, None]):
    # Entry of this phase
    def transform(self, program: Program) -> TACProg:
        labelManager = LabelManager()
        tacFuncs = []
        tacVars = []

        for var in program.declarations():
            init = False
            value = 0
            if var.init_expr != NULL:
                if isinstance(var.init_expr, IntLiteral):
                    value = var.init_expr.value
                    init = True
                elif isinstance(var.init_expr, ArrayInit):
                    assert isinstance(var.var_t.type, ArrayType)
                    value = [expr.value for expr in var.init_expr]
                    value += [0] * (var.var_t.type.length - len(value))
                    init = True
            tacVars.append(TACVar(var.symbol.name, value, init, var.symbol.type.size))

        for funcName, astFunc in program.functions().items():
            # in step9, you need to use real parameter count
            emitter = TACFuncEmitter(
                FuncLabel(funcName), len(astFunc.params), labelManager
            )
            for param in astFunc.params:
                param.accept(self, emitter)
            astFunc.body.accept(self, emitter)
            tacFuncs.append(emitter.visitEnd())
        return TACProg(tacFuncs, tacVars)

    def visitBlock(self, block: Block, mv: TACFuncEmitter) -> None:
        for child in block:
            child.accept(self, mv)

    def visitReturn(self, stmt: tree.Return, mv: TACFuncEmitter) -> None:
        stmt.expr.accept(self, mv)
        mv.visitReturn(stmt.expr.getattr("val"))

    def visitBreak(self, stmt: Break, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getBreakLabel())

    def visitContinue(self, stmt: Continue, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getContinueLabel())

    def visitIdentifier(self, ident: Identifier, mv: TACFuncEmitter) -> None:
        """
        1. Set the 'val' attribute of ident as the temp variable of the 'symbol' attribute of ident.
        """
        if ident.symbol.isGlobal:
            addr = mv.visitGlobalVar(ident.symbol)
            ident.symbol.addr = addr
            if mv.right:
                if isinstance(ident.symbol.type, ArrayType):
                    ident.setattr("val", ident.symbol.addr)
                else:
                    ident.setattr("val", mv.visitAddr(ident.symbol.addr))
        elif mv.right and isinstance(ident.symbol.type, ArrayType):
            ident.setattr("val", ident.symbol.addr)
        else:
            if mv.right:
                ident.setattr("val", ident.symbol.temp)

    def visitParameter(self, param: tree.Parameter, mv: TACFuncEmitter) -> None:
        symbol = param.symbol
        if isinstance(symbol.type, ArrayType):
            addr = mv.freshTemp()
            symbol.addr = addr
        else:
            temp = mv.freshTemp()
            symbol.temp = temp

    def visitArrayInit(self, init: ArrayInit, mv: TACFuncEmitter) -> None:
        for child in init.children:
            child.accept(self, mv)

    def visitDeclaration(self, decl: Declaration, mv: TACFuncEmitter) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.visitAssignment to set it.
        """
        symbol = decl.symbol
        if isinstance(decl, ArrayDeclaration):
            addr = mv.visitAlloc(symbol.type.size)
            symbol.addr = addr
            if decl.init_expr is not NULL:
                decl.init_expr.accept(self, mv)
                temps = [expr.getattr("val") for expr in decl.init_expr]
                mv.visitArrayInit(addr, symbol.type.size, temps)
        else:
            temp = mv.freshTemp()
            symbol.temp = temp
            if decl.init_expr is not NULL:
                decl.init_expr.accept(self, mv)
                mv.visitAssignment(temp, decl.init_expr.getattr("val"))

    def visitArrayIndex(self, idx: ArrayIndex, mv: TACFuncEmitter) -> None:
        # save father command
        right = mv.right
        # we only need left value of base
        mv.right = False
        idx.base.accept(self, mv)
        # we need right value of index
        mv.right = True
        idx.index.accept(self, mv)
        base_symbol = idx.base.symbol
        addr = mv.emitBinary(
            tacop.TacBinaryOp.ADD,
            base_symbol.addr,
            mv.emitBinary(
                tacop.TacBinaryOp.MUL,
                idx.index.getattr("val"),
                mv.visitLoadImm(idx.symbol.type.size),
            ),
        )
        idx.symbol.addr = addr
        if right:
            if isinstance(idx.symbol.type, ArrayType):
                raise NotImplementedError("Don't support array assigning currently")
            idx.setattr("val", mv.visitAddr(addr))

    def visitAssignment(self, expr: Assignment, mv: TACFuncEmitter) -> None:
        """
        1. Visit the right hand side of expr, and get the temp variable of left hand side.
        2. Use mv.visitAssignment to emit an assignment instruction.
        3. Set the 'val' attribute of expr as the value of assignment instruction.
        """
        expr.rhs.accept(self, mv)
        mv.right = False
        expr.lhs.accept(self, mv)
        mv.right = True
        symbol = expr.lhs.symbol
        if isinstance(symbol.type, ArrayType):
            raise NotImplementedError("Don't support array assigning currently")
        if hasattr(symbol, "addr"):
            result = mv.visitAddrAssignment(symbol.addr, expr.rhs.getattr("val"))
        else:
            result = mv.visitAssignment(symbol.temp, expr.rhs.getattr("val"))
        expr.setattr("val", result)

    def visitIf(self, stmt: If, mv: TACFuncEmitter) -> None:
        stmt.cond.accept(self, mv)

        if stmt.otherwise is NULL:
            skipLabel = mv.freshLabel()
            mv.emitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitLabel(skipLabel)
        else:
            skipLabel = mv.freshLabel()
            exitLabel = mv.freshLabel()
            mv.emitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitBranch(exitLabel)
            mv.visitLabel(skipLabel)
            stmt.otherwise.accept(self, mv)
            mv.visitLabel(exitLabel)

    def visitWhile(self, stmt: While, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.emitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitFor(self, stmt: For, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        stmt.init.accept(self, mv)
        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.emitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        stmt.after.accept(self, mv)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitCall(self, call: tree.Call, mv: TACFuncEmitter) -> None:
        for arg in call.args:
            arg.accept(self, mv)
        args = [arg.getattr("val") for arg in call.args]
        call.setattr("val", mv.emitCall(FuncLabel(call.ident.value), args))

    def visitUnary(self, expr: tree.Unary, mv: TACFuncEmitter) -> None:
        expr.operand.accept(self, mv)

        op = {
            node.UnaryOp.Neg: tacop.TacUnaryOp.NEG,
            node.UnaryOp.LogicNot: tacop.TacUnaryOp.LOGIC_NOT,
            node.UnaryOp.BitNot: tacop.TacUnaryOp.BIT_NOT,
            # You can add unary operations here.
        }[expr.op]
        expr.setattr("val", mv.emitUnary(op, expr.operand.getattr("val")))

    def visitBinary(self, expr: tree.Binary, mv: TACFuncEmitter) -> None:
        expr.lhs.accept(self, mv)
        expr.rhs.accept(self, mv)

        op = {
            node.BinaryOp.Add: tacop.TacBinaryOp.ADD,
            node.BinaryOp.LogicOr: tacop.TacBinaryOp.OR,
            node.BinaryOp.Sub: tacop.TacBinaryOp.SUB,
            node.BinaryOp.Mul: tacop.TacBinaryOp.MUL,
            node.BinaryOp.Div: tacop.TacBinaryOp.DIV,
            node.BinaryOp.Mod: tacop.TacBinaryOp.MOD,
            node.BinaryOp.LogicAnd: tacop.TacBinaryOp.AND,
            node.BinaryOp.EQ: tacop.TacBinaryOp.EQU,
            node.BinaryOp.NE: tacop.TacBinaryOp.NEQ,
            node.BinaryOp.LT: tacop.TacBinaryOp.SLT,
            node.BinaryOp.LE: tacop.TacBinaryOp.LEQ,
            node.BinaryOp.GT: tacop.TacBinaryOp.SGT,
            node.BinaryOp.GE: tacop.TacBinaryOp.GEQ,
            # You can add binary operations here.
        }[expr.op]
        expr.setattr(
            "val", mv.emitBinary(op, expr.lhs.getattr("val"), expr.rhs.getattr("val"))
        )

    def visitCondExpr(self, expr: ConditionExpression, mv: TACFuncEmitter) -> None:
        """
        1. Refer to the implementation of visitIf and visitBinary.
        """
        expr.cond.accept(self, mv)
        temp = mv.freshTemp()

        skipLabel = mv.freshLabel()
        exitLabel = mv.freshLabel()
        mv.emitCondBranch(tacop.CondBranchOp.BEQ, expr.cond.getattr("val"), skipLabel)
        expr.then.accept(self, mv)
        mv.visitAssignment(temp, expr.then.getattr("val"))
        mv.visitBranch(exitLabel)
        mv.visitLabel(skipLabel)
        expr.otherwise.accept(self, mv)
        mv.visitAssignment(temp, expr.otherwise.getattr("val"))
        mv.visitLabel(exitLabel)
        expr.setattr("val", temp)

    def visitIntLiteral(self, expr: IntLiteral, mv: TACFuncEmitter) -> None:
        expr.setattr("val", mv.visitLoadImm(expr.value))
