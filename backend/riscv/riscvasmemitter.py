from backend.asmemitter import AsmEmitter
from utils.error import IllegalArgumentException
from utils.label.label import LabelKind
from utils.riscv import Riscv, RvBinaryOp, RvUnaryOp
from utils.tac.reg import Imm
from utils.tac.tacfunc import TACFunc
from utils.tac.tacinstr import *
from utils.tac.tacvar import TACVar
from utils.tac.tacvisitor import TACVisitor

from ..subroutineemitter import SubroutineEmitter
from ..subroutineinfo import SubroutineInfo

"""
RiscvAsmEmitter: an AsmEmitter for RiscV
"""


class RiscvAsmEmitter(AsmEmitter):
    def __init__(
            self,
            allocatableRegs: list[Reg],
            callerSaveRegs: list[Reg],
            calleeSaveRegs: list[Reg],
    ) -> None:
        super().__init__(allocatableRegs, callerSaveRegs, calleeSaveRegs)
        # the start of the asm code
        # int step10, you need to add the declaration of global var here

    # transform tac instrs to RiscV instrs
    # collect some info which is saved in SubroutineInfo for SubroutineEmitter
    def selectInstr(self, func: TACFunc) -> tuple[list[TACInstr], SubroutineInfo]:

        selector: RiscvAsmEmitter.RiscvInstrSelector = (
            RiscvAsmEmitter.RiscvInstrSelector(func.entry)
        )
        for instr in func.getInstrSeq():
            instr.accept(selector)

        info = SubroutineInfo(func.entry)

        return selector.seq, info

    # use info to construct a RiscvSubroutineEmitter
    def emitSubroutine(self, info: SubroutineInfo):
        return RiscvSubroutineEmitter(self, info)

    # return all the string stored in asm code printer
    def emitEnd(self):
        return self.printer.close()

    class RiscvInstrSelector(TACVisitor):
        def __init__(self, entry: Label) -> None:
            self.entry = entry
            self.seq = []

        def visitOther(self, instr: TACInstr) -> None:
            raise NotImplementedError("RiscvInstrSelector visit{} not implemented".format(type(instr).__name__))

        # in step11, you need to think about how to deal with globalTemp in almost all the visit functions.
        def visitReturn(self, instr: Return) -> None:
            if instr.value is not None:
                self.seq.append(Riscv.Move(Riscv.A0, instr.value))
            else:
                self.seq.append(Riscv.LoadImm(Riscv.A0, 0))
            self.seq.append(Riscv.JumpToEpilogue(self.entry))

        def visitMark(self, instr: Mark) -> None:
            self.seq.append(Riscv.RiscvLabel(instr.label))

        def visitLoadImm4(self, instr: LoadImm4) -> None:
            self.seq.append(Riscv.LoadImm(instr.dst, instr.value))

        def visitLoadParams(self, instr: LoadParams) -> None:
            self.seq.append(Riscv.LoadParams(instr.dsts))

        def visitCall(self, instr: Call) -> None:
            self.seq.append(Riscv.Call(instr.func, instr.ret, instr.args))

        def visitLoadSymbol(self, instr: LoadSymbol) -> None:
            self.seq.append(Riscv.LoadSymbol(instr.dst, instr.name))

        def visitLoad(self, instr: Load) -> None:
            self.seq.append(Riscv.LoadAddr(instr.dst, instr.src))

        def visitAlloc(self, instr: Alloc) -> None:
            self.seq.append(Riscv.Alloc(instr.dst, instr.size))

        def visitMemset(self, instr: Memset) -> None:
            self.seq.append(Riscv.Call(Riscv.MEMSET, Riscv.ZERO, [instr.addr, Riscv.ZERO, instr.size]))

        def visitUnary(self, instr: Unary) -> None:
            op = {
                TacUnaryOp.NEG: RvUnaryOp.NEG,
                TacUnaryOp.LOGIC_NOT: RvUnaryOp.SEQZ,
                TacUnaryOp.BIT_NOT: RvUnaryOp.NOT,
                # You can add unary operations here.
            }[instr.op]
            self.seq.append(Riscv.Unary(op, instr.dst, instr.operand))

        def visitBinary(self, instr: Binary) -> None:
            """
            For different tac operation, you should translate it to different RiscV code
            A tac operation may need more than one RiscV instruction
            """
            if instr.op == TacBinaryOp.OR:
                self.seq.append(Riscv.Binary(RvBinaryOp.OR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.AND:
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.lhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, Riscv.ZERO, instr.dst))
                self.seq.append(Riscv.Binary(RvBinaryOp.AND, instr.dst, instr.dst, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.EQU:
                self.seq.append(Riscv.Binary(RvBinaryOp.XOR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.SLTIU, instr.dst, instr.dst, Imm(1)))
            elif instr.op == TacBinaryOp.NEQ:
                self.seq.append(Riscv.Binary(RvBinaryOp.XOR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.SLTU, instr.dst, Riscv.ZERO, instr.dst))
            elif instr.op == TacBinaryOp.SGT:
                self.seq.append(Riscv.Binary(RvBinaryOp.SLT, instr.dst, instr.rhs, instr.lhs))
            elif instr.op == TacBinaryOp.GEQ:
                self.seq.append(Riscv.Binary(RvBinaryOp.SLT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.XOR, instr.dst, instr.dst, Imm(1)))
            elif instr.op == TacBinaryOp.LEQ:
                self.seq.append(Riscv.Binary(RvBinaryOp.SLT, instr.dst, instr.rhs, instr.lhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.XOR, instr.dst, instr.dst, Imm(1)))
            else:
                op = {
                    TacBinaryOp.ADD: RvBinaryOp.ADD,
                    TacBinaryOp.SUB: RvBinaryOp.SUB,
                    TacBinaryOp.MUL: RvBinaryOp.MUL,
                    TacBinaryOp.DIV: RvBinaryOp.DIV,
                    TacBinaryOp.MOD: RvBinaryOp.REM,
                    TacBinaryOp.SLT: RvBinaryOp.SLT,
                    # You can add binary operations here.
                }[instr.op]
                self.seq.append(Riscv.Binary(op, instr.dst, instr.lhs, instr.rhs))

        def visitCondBranch(self, instr: CondBranch) -> None:
            self.seq.append(Riscv.Branch(instr.cond, instr.label))

        def visitBranch(self, instr: Branch) -> None:
            self.seq.append(Riscv.Jump(instr.target))

        def visitAssign(self, instr: Assign) -> None:
            self.seq.append(Riscv.Move(instr.dst, instr.src))

        def visitAddrAssign(self, instr: AddrAssign) -> None:
            self.seq.append(Riscv.StoreWord(instr.src, instr.addr, instr.offset))

        # in step9, you need to think about how to pass the parameters and how to store and restore callerSave regs
        # in step11, you need to think about how to store the array

    def emitMemsetFunc(self):
        self.printer.printLabel(Riscv.MEMSET)
        self.printer.printInstr(Riscv.LoadImm(Riscv.A4, 4))
        self.printer.printInstr(Riscv.Binary(RvBinaryOp.MUL, Riscv.A2, Riscv.A2, Riscv.A4))
        self.printer.printInstr(Riscv.Binary(RvBinaryOp.ADD, Riscv.A3, Riscv.A0, Riscv.A2))
        self.printer.printLabel(Label(LabelKind.TEMP, "memset_loop"))
        self.printer.printInstr(Riscv.Branch(Riscv.A3, Label(LabelKind.TEMP, "memset_end"), Riscv.A0))
        self.printer.printInstr(Riscv.Binary(RvBinaryOp.SUB, Riscv.A3, Riscv.A3, Riscv.A4))
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.A1, Riscv.A3, 0))
        self.printer.printInstr(Riscv.Jump(Label(LabelKind.TEMP, "memset_loop")))
        self.printer.printLabel(Label(LabelKind.TEMP, "memset_end"))
        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")

    def emitGlobalVars(self, variables: List[TACVar]):
        self.printer.printComment("COMPILED BY ZHC")
        self.emitMemsetFunc()
        self.printer.printComment("GlOBAL VAR")
        self.printer.printSection("data")
        for var in variables:
            if not var.initialized:
                continue
            self.printer.printSection("globl", var.name)
            self.printer.printLabel(Label(LabelKind.TEMP, var.name))
            self.printer.println(f".word {var.value}")
        self.printer.println("")

        self.printer.printSection("bss")
        for var in variables:
            if var.initialized:
                continue
            self.printer.printSection("globl", var.name)
            self.printer.printLabel(Label(LabelKind.TEMP, var.name))
            self.printer.println(f".space {var.size}")
        self.printer.println("")

        self.printer.printSection("text")
        self.printer.printSection("global", "main")
        self.printer.println("")


"""
RiscvAsmEmitter: an SubroutineEmitter for RiscV
"""


class RiscvSubroutineEmitter(SubroutineEmitter):
    def __init__(self, emitter: RiscvAsmEmitter, info: SubroutineInfo) -> None:
        super().__init__(emitter, info)

        # + 8 is for the RA and FP reg
        self.nextLocalOffset = 4 * len(Riscv.CalleeSaved) + 8

        # the buf which stored all the NativeInstrs in this function
        self.buf: list[NativeInstr] = []

        # from temp to int
        # record where a temp is stored in the stack
        self.offsets = {}

        self.printer.printLabel(info.funcLabel)

        # in step9, step11 you can compute the offset of local array and parameters here
        self.nextParamOffset = 0
        self.param_buf = []

    def emitComment(self, comment: str) -> None:
        # you can add some log here to help you debug
        pass

    def readParam(self, params: List[Temp]) -> None:
        for i, param in enumerate(params):
            self.offsets[param.index] = self.nextLocalOffset + 4 * i

    def prepareParam(self, src: Reg) -> None:
        self.param_buf.append(Riscv.NativeStoreWord(src, Riscv.SP, self.nextParamOffset))
        self.nextParamOffset += 4

    def alloc(self, dst: Reg, size: int) -> None:
        self.buf.append(Riscv.LoadImm(dst, self.nextLocalOffset).toNative([dst], []))
        self.buf.append(Riscv.Binary(RvBinaryOp.ADD, dst, dst, Riscv.SP).toNative([dst], [dst, Riscv.SP]))
        self.nextLocalOffset += size

    def beforeCall(self):
        if self.nextParamOffset > 0:
            self.buf.append(Riscv.SPAdd(-self.nextParamOffset))
            self.buf.extend(self.param_buf)
            self.param_buf = []

    def afterCall(self) -> None:
        if self.nextParamOffset > 0:
            self.buf.append(Riscv.SPAdd(self.nextParamOffset))
            self.nextParamOffset = 0

    # store some temp to stack
    # usually happen when reaching the end of a basic block
    # in step9, you need to think about the function parameters here
    def emitStoreToStack(self, src: Reg) -> None:
        if src.temp.index not in self.offsets:
            self.offsets[src.temp.index] = self.nextLocalOffset
            self.nextLocalOffset += 4
        self.buf.append(
            Riscv.NativeStoreWord(src, Riscv.SP, self.offsets[src.temp.index])
        )

    # load some temp from stack
    # usually happen when using a temp which is stored to stack before
    # in step9, you need to think about the function parameters here
    def emitLoadFromStack(self, dst: Reg, src: Temp):
        if src.index not in self.offsets:
            raise IllegalArgumentException()
        else:
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index])
            )

    # add a NativeInstr to buf
    # when calling the function emitEnd, all the instr in buf will be transformed to RiscV code
    def emitNative(self, instr: NativeInstr):
        self.buf.append(instr)

    def emitLabel(self, label: Label):
        self.buf.append(Riscv.RiscvLabel(label).toNative([], []))

    def emitEnd(self):
        self.printer.printComment("start of prologue")

        # save FP reg
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.FP, Riscv.SP,
                                                      4 * len(Riscv.CalleeSaved) + 4 - self.nextLocalOffset))
        # set FP reg
        self.printer.printInstr(Riscv.Move(Riscv.FP, Riscv.SP))

        self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset))

        # in step9, you need to think about how to store RA here
        # you can get some ideas from how to save CalleeSaved regs
        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeStoreWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved)))

        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        # in step9, you need to think about how to pass the parameters here
        # you can use the stack or regs

        # using asm code printer to output the RiscV code
        for instr in self.buf:
            self.printer.printInstr(instr)

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(
            Label(LabelKind.TEMP, self.info.funcLabel.name + Riscv.EPILOGUE_SUFFIX)
        )
        self.printer.printComment("start of epilogue")

        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeLoadWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )

        # resume FP and RA reg
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.FP, Riscv.SP, 4 * len(Riscv.CalleeSaved) + 4))
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved)))

        self.printer.printInstr(Riscv.SPAdd(self.nextLocalOffset))
        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")
