"""Dead-code elimination over pruned SSA form.

The entry point :func:`eliminate_dead_code` takes the SSA :class:`Module`
returned by :func:`to_ssa` and returns a brand new SSA :class:`Module` with
the dead definitions removed; the input module is never mutated and shares
no mutable function, block, instruction or phi containers with the result.

Liveness is traced per function from the observable roots:

* the operands of every :class:`Return` and :class:`Branch`;
* every :class:`Call` instruction, whose result may be unused -- the call
  itself and its position in the instruction stream are observable
  behaviour, so the instruction is always kept and its argument definitions
  become live in turn.

From those roots the analysis walks def/use chains: a live :class:`Const`,
:class:`BinOp` or :class:`Phi` forces its operands live, and so on until a
fix point.  Chains referenced only by other dead definitions and closed
phi-only cycles are never seeded and disappear together.  Constant folding,
branch rewriting, block deletion and control-flow reordering are out of
scope: every block and terminator is copied verbatim, surviving
instructions keep their relative order, surviving phis keep their
predecessor order, and SSA numbering is left untouched (holes are fine).

Passing a non-:class:`Module` raises :class:`TypeError`; passing an SSA
module to :func:`to_ssa` already yields an independent copy, and an empty
module or one with nothing to delete likewise comes back as an independent
copy.
"""
from .ir_nodes import (
    BinOp,
    Block,
    Branch,
    Call,
    Const,
    Copy,
    Function,
    Jump,
    Module,
    Parameter,
    Phi,
    Return,
    Temp,
)


def eliminate_dead_code(module: Module) -> Module:
    """Return an SSA copy of ``module`` with dead definitions removed.

    The input module is left untouched and no mutable containers are shared
    with the result.  Eliminating an already-pruned module yields a
    structurally and textually equivalent copy.

    :raises TypeError: if ``module`` is not a :class:`Module` instance.
    :raises ValueError: if ``module`` is not in SSA form.
    """
    if not isinstance(module, Module):
        raise TypeError(
            f"eliminate_dead_code expects a Module, got "
            f"{type(module).__name__}"
        )
    if not getattr(module, "ssa", False):
        raise ValueError(
            "eliminate_dead_code expects an SSA Module; convert the module "
            "with to_ssa first"
        )
    return Module(
        [_DeadCodeEliminator(func).run() for func in module.functions],
        ssa=True,
    )


# --------------------------------------------------------------------------
# Operand helpers (mirror ssa.py; work uniformly on instructions, phis and
# terminators)
# --------------------------------------------------------------------------


def _instruction_operands(ins):
    if isinstance(ins, Copy):
        return [ins.src]
    if isinstance(ins, BinOp):
        return [ins.left, ins.right]
    if isinstance(ins, Call):
        return list(ins.args)
    return []


def _terminator_operands(term):
    if isinstance(term, Return) and term.value is not None:
        return [term.value]
    if isinstance(term, Branch):
        return [term.cond]
    return []


# --------------------------------------------------------------------------
# Per-function elimination
# --------------------------------------------------------------------------


class _DeadCodeEliminator:
    def __init__(self, func: Function):
        self.func = func

    def run(self) -> Function:
        # Deep copy first; the copy is what liveness analysis reads, so the
        # input function is never touched and no block, instruction or phi
        # object is shared with the result.
        new_func = _clone_ssa_function(self.func)

        live = self._live_definitions(new_func)

        for block in new_func.blocks:
            block.phis = [
                phi for phi in block.phis if id(phi.dest) in live
            ]
            block.instructions = [
                ins for ins in block.instructions
                if isinstance(ins, Call) or id(ins.dest) in live
            ]

        return new_func

    # -- liveness ----------------------------------------------------------

    def _live_definitions(self, func: Function) -> set[int]:
        """Return the ``id()`` set of every definition kept alive.

        Roots are the terminator operands and every call (with its args).
        A worklist walks the definitions behind each live operand through
        phis and ordinary instructions until nothing new is marked.
        """
        definer: dict[int, object] = {}
        for block in func.blocks:
            for phi in block.phis:
                definer[id(phi.dest)] = phi
            for ins in block.instructions:
                definer[id(ins.dest)] = ins

        worklist: list[Temp] = []
        for block in func.blocks:
            for ins in block.instructions:
                if isinstance(ins, Call):
                    worklist.extend(ins.args)
            worklist.extend(_terminator_operands(block.terminator))

        live: set[int] = set()
        while worklist:
            value = worklist.pop()
            key = id(value)
            if key in live:
                continue
            live.add(key)
            site = definer.get(key)
            if isinstance(site, Phi):
                # Parameters and values without a local definition have no
                # operands to trace further.
                worklist.extend(site.entries.values())
            elif site is not None:
                worklist.extend(_instruction_operands(site))
        return live


# --------------------------------------------------------------------------
# Structural SSA clone (same discipline as ssa._clone_ssa_function)
# --------------------------------------------------------------------------


def _clone_ssa_function(func: Function) -> Function:
    new_blocks = [Block(block.id) for block in func.blocks]
    block_map = dict(zip(func.blocks, new_blocks))

    number: dict[Temp, Temp] = {}

    def assign(temp: Temp) -> Temp:
        mapped = number.get(temp)
        if mapped is None:
            # Preserve the original ids: eliminating definitions may leave
            # holes, but surviving values keep their numbers.
            mapped = Temp(temp.id, temp.type)
            number[temp] = mapped
        return mapped

    new_params: list[Parameter] = []
    for param in func.params:
        new_params.append(
            Parameter(param.name, param.slot, assign(param.temp))
        )
    for block in func.blocks:
        for phi in block.phis:
            assign(phi.dest)
        for ins in block.instructions:
            assign(ins.dest)

    def ref(value):
        return number[value] if value is not None else None

    for block, new_block in zip(func.blocks, new_blocks):
        for phi in block.phis:
            entries = {
                block_map[pred]: ref(value)
                for pred, value in sorted(
                    phi.entries.items(), key=lambda item: item[0].id
                )
            }
            new_block.phis.append(Phi(ref(phi.dest), entries))
        for ins in block.instructions:
            if isinstance(ins, Const):
                clone = Const(ref(ins.dest), ins.value)
            elif isinstance(ins, Copy):
                clone = Copy(ref(ins.dest), ref(ins.src))
            elif isinstance(ins, BinOp):
                clone = BinOp(
                    ref(ins.dest), ins.operator, ref(ins.left),
                    ref(ins.right), ins.kind, ins.type,
                )
            else:
                clone = Call(
                    ref(ins.dest), ins.name,
                    [ref(arg) for arg in ins.args], ins.type,
                )
            new_block.instructions.append(clone)
        term = block.terminator
        if isinstance(term, Return):
            new_block.terminator = Return(ref(term.value))
        elif isinstance(term, Jump):
            new_block.terminator = Jump(block_map[term.target])
        elif isinstance(term, Branch):
            new_block.terminator = Branch(
                ref(term.cond),
                block_map[term.true_target],
                block_map[term.false_target],
            )

    return Function(
        func.name,
        new_params,
        func.ret_type,
        list(func.locals),
        new_blocks,
        block_map[func.entry],
        ssa=True,
    )
