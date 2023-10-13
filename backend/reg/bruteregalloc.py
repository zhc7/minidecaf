import random

from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.cfg import CFG
from backend.dataflow.loc import Loc
from backend.reg.regalloc import RegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from backend.subroutineemitter import SubroutineEmitter
from backend.subroutineinfo import SubroutineInfo
from utils.riscv import Riscv
from utils.tac.reg import Reg
from utils.tac.tacop import InstrKind
from utils.tac.temp import Temp

"""
BruteRegAlloc: one kind of RegAlloc

bindings: map from temp.index to Reg

we don't need to take care of GlobalTemp here
because we can remove all the GlobalTemp in selectInstr process

1. accept：根据每个函数的 CFG 进行寄存器分配，寄存器分配结束后生成相应汇编代码
2. bind：将一个 Temp 与寄存器绑定
3. unbind：将一个 Temp 与相应寄存器解绑定
4. localAlloc：根据数据流对一个 BasicBlock 内的指令进行寄存器分配
5. allocForLoc：每一条指令进行寄存器分配
6. allocRegFor：根据数据流决定为当前 Temp 分配哪一个寄存器
"""


class BruteRegAlloc(RegAlloc):
    def __init__(self, emitter: RiscvAsmEmitter) -> None:
        super().__init__(emitter)
        self.bindings = {}
        self.reservations = {}
        for reg in emitter.allocatableRegs:
            reg.used = False

    def accept(self, graph: CFG, info: SubroutineInfo) -> None:
        subEmitter = self.emitter.emitSubroutine(info)
        for bb in graph.iterator():
            # you need to think more here
            # maybe we don't need to alloc regs for all the basic blocks
            if not graph.isReachable(bb):
                continue
            if bb.label is not None:
                subEmitter.emitLabel(bb.label)
            self.localAlloc(bb, subEmitter)
        subEmitter.emitEnd()

    def reserve(self, temp: Temp, reg: Reg):
        if reg in self.reservations:
            self.reservations[reg].append(temp.index)
        else:
            self.reservations[reg] = [temp.index]

    def bind(self, temp: Temp, reg: Reg):
        reg.used = True
        self.bindings[temp.index] = reg
        reg.occupied = True
        reg.temp = temp

    def unbind(self, temp: Temp):
        if temp.index in self.bindings:
            self.bindings[temp.index].occupied = False
            self.bindings.pop(temp.index)

    def localAlloc(self, bb: BasicBlock, subEmitter: SubroutineEmitter):
        self.bindings.clear()
        self.reservations.clear()
        for reg in self.emitter.allocatableRegs:
            reg.occupied = False

        if len(bb.locs) > 0:
            loc = bb.locs[0]
            if isinstance(loc.instr, Riscv.LoadParams):
                for i, param in enumerate(loc.instr.dsts):
                    if i < 8:
                        self.bind(param, Riscv.ArgRegs[i])
                    else:
                        break
                subEmitter.readParam(loc.instr.dsts[8:])

        for loc in bb.allSeq():
            if isinstance(loc.instr, Riscv.Call):
                instr = loc.instr
                for i in range(min(len(instr.srcs), 8)):
                    self.reserve(instr.srcs[i], Riscv.ArgRegs[i])

        # in step9, you may need to think about how to store caller save regs here
        for loc in bb.allSeq():
            subEmitter.emitComment(str(loc.instr))

            if isinstance(loc.instr, Riscv.LoadParams):
                continue
            if isinstance(loc.instr, Riscv.Call):
                self.allocForCall(loc, subEmitter)
            else:
                self.allocForLoc(loc, subEmitter)
        for tempindex in bb.liveOut:
            if tempindex in self.bindings:
                subEmitter.emitStoreToStack(self.bindings.get(tempindex))

        if (not bb.isEmpty()) and (bb.kind is not BlockKind.CONTINUOUS):
            self.allocForLoc(bb.locs[len(bb.locs) - 1], subEmitter)

    def spill(self, reg: Reg, subEmitter: SubroutineEmitter):
        subEmitter.emitStoreToStack(reg)
        subEmitter.emitComment("  spill {} ({})".format(str(reg), str(reg.temp)))
        self.unbind(reg.temp)

    def allocForLoc(self, loc: Loc, subEmitter: SubroutineEmitter):
        instr = loc.instr
        srcRegs: list[Reg] = []
        dstRegs: list[Reg] = []

        for i in range(len(instr.srcs)):
            temp = instr.srcs[i]
            if isinstance(temp, Reg):
                srcRegs.append(temp)
            else:
                srcRegs.append(self.allocRegFor(temp, True, loc.liveIn, subEmitter))

        for i in range(len(instr.dsts)):
            temp = instr.dsts[i]
            if isinstance(temp, Reg):
                dstRegs.append(temp)
            else:
                dstRegs.append(self.allocRegFor(temp, False, loc.liveIn, subEmitter))

        if isinstance(instr, Riscv.Alloc):
            subEmitter.alloc(dstRegs[0], instr.size)
        else:
            subEmitter.emitNative(instr.toNative(dstRegs, srcRegs))

    def allocForCall(self, loc: Loc, subEmitter: SubroutineEmitter):
        instr = loc.instr
        # 1. prepare params
        for i in range(min(len(instr.srcs), 8)):
            dst = Riscv.ArgRegs[i]
            src = instr.srcs[i]
            if isinstance(src, Reg):
                subEmitter.emitNative(Riscv.Move(dst, src).toNative([dst], [src]))
            elif dst.temp != src:
                if dst.occupied and dst.temp.index in loc.liveIn:
                    self.spill(dst, subEmitter)
                subEmitter.emitLoadFromStack(dst, instr.srcs[i])
            self.reservations[dst].pop(0)
        for i in range(8, len(instr.srcs)):
            src = instr.srcs[i]
            if src.index in self.bindings:
                src = self.bindings[src.index]
            if not isinstance(src, Reg):
                for reg in self.emitter.allocatableRegs:
                    if reg not in Riscv.ArgRegs and (not reg.occupied or reg.temp.index not in loc.liveIn):
                        subEmitter.emitLoadFromStack(reg, src)
                        self.bind(src, reg)
                        src = reg
                        break
            subEmitter.prepareParam(src)

        # 2. save callerSave regs
        for reg in Riscv.CallerSaved:
            if reg.occupied and reg.temp.index in loc.liveOut:
                subEmitter.emitStoreToStack(reg)

        # 3. call
        subEmitter.beforeCall()
        subEmitter.emitNative(instr.toNative([], []))
        subEmitter.afterCall()

        # 4. get return value
        assert len(instr.dsts) <= 1
        bind2A0 = False
        if len(instr.dsts) == 1:
            dst = instr.dsts[0]
            if dst.index in self.bindings:
                dst = self.bindings[dst.index]
            if isinstance(dst, Reg):
                subEmitter.emitNative(Riscv.Move(dst, Riscv.A0).toNative([dst], [Riscv.A0]))
            elif not Riscv.A0.occupied:
                bind2A0 = True
            else:
                dst = self.allocRegFor(dst, False, loc.liveIn, subEmitter)
                subEmitter.emitNative(Riscv.Move(dst, Riscv.A0).toNative([dst], [Riscv.A0]))

        # 5. restore callerSave regs
        for reg in Riscv.CallerSaved:
            if reg.occupied and reg.temp.index in loc.liveOut:
                subEmitter.emitLoadFromStack(reg, reg.temp)

        if bind2A0:
            self.bind(instr.dsts[0], Riscv.A0)

    def allocRegFor(
        self, temp: Temp, isRead: bool, live: set[int], subEmitter: SubroutineEmitter
    ):
        if temp.index in self.bindings:
            return self.bindings[temp.index]

        top_reservations = {line[0]: reg for reg, line in self.reservations.items() if len(line) > 0}
        if temp.index in top_reservations:
            assert not isRead
            reg = top_reservations[temp.index]
            if reg.occupied and reg.temp.index in live:
                self.spill(reg, subEmitter)
            self.bind(temp, reg)
            return reg

        unreserved = [reg for reg in self.emitter.callerSaveRegs if reg not in top_reservations.values()]
        precedence = [self.emitter.calleeSaveRegs, unreserved, top_reservations.keys()]

        for regs in precedence:
            for reg in regs:
                if (not reg.occupied) or (reg.temp.index not in live):
                    subEmitter.emitComment(
                        "  allocate {} to {}  (read: {}):".format(
                            str(temp), str(reg), str(isRead)
                        )
                    )
                    if isRead:
                        subEmitter.emitLoadFromStack(reg, temp)
                    if reg.occupied:
                        self.unbind(reg.temp)
                    self.bind(temp, reg)
                    return reg

        reg = self.emitter.allocatableRegs[
            random.randint(0, len(self.emitter.allocatableRegs) - 1)
        ]
        self.spill(reg, subEmitter)
        self.bind(temp, reg)
        subEmitter.emitComment(
            "  allocate {} to {} (read: {})".format(str(temp), str(reg), str(isRead))
        )
        if isRead:
            subEmitter.emitLoadFromStack(reg, temp)
        return reg
