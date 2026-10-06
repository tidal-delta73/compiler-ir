"""Dead-code elimination on pruned SSA form.

The entry point :func:`eliminate_dead_code` takes an SSA :class:`Module`
produced by :func:`~compiler_ir.ssa.to_ssa` and returns a brand new SSA
:class:`Module` with unreachable definitions removed; the input module is
never mutated and shares no mutable function, block, instruction or phi
container with the result.

Liveness is traced per function, definition-to-use, from fixed roots:

* every operand of a ``Return`` or ``Branch`` terminator;
* every ``Call`` instruction, regardless of whether its result is used --
  the call itself and its position in the instruction stream are observable
  behavior -- together with all of its argument definitions.

A ``Const``, ``BinOp`` or ``Phi`` that cannot reach one of those roots is
deleted.  The trace runs on the whole function at once, so chains feeding
only other dead definitions and closed phi cycles die together in one pass;
elimination is then repeated until no new dead definition appears.

The pass is purely subtractive.  No constants are folded, branches
rewritten, blocks deleted or control flow reordered.  Function order,
signatures, parameters, return types, block labels and order, terminators,
the relative order of surviving instructions and the predecessor order of
surviving phis are all preserved, and SSA numbers of surviving values are
left untouched (the output may therefore contain numbering holes).

Applying the pass to an already minimal module returns an independent,
structurally and textually equivalent copy, so a second
:func:`eliminate_dead_code` changes nothing.  A non-:class:`Module` object
raises :class:`TypeError`; a non-SSA :class:`Module` raises
:class:`ValueError`.
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
    """Return a new SSA :class:`Module` with dead definitions removed.

    The input module is left untouched; the result shares no mutable
    function, block, instruction or phi object with it.

    :raises TypeError: if ``module`` is not a :class:`Module` instance.
    :raises ValueError: if ``module`` is not in SSA form.
    """
    if not isinstance(module, Module):
        raise TypeError(
            "eliminate_dead_code expects a Module, got "
            f"{type(module).__name__}"
        )
    if not getattr(module, "ssa", False):
        raise ValueError("eliminate_dead_code expects an SSA Module")
    return Module(
        [_eliminate_in_function(func) for func in module.functions], ssa=True
    )


# --------------------------------------------------------------------------
# Operand helpers (mirror the ones used by the SSA conversion)
# --------------------------------------------------------------------------


def _operands(ins):
    if isinstance(ins, BinOp):
        return [ins.left, ins.right]
    if isinstance(ins, Call):
        return list(ins.args)
    # Const has no operands; a Copy cannot survive the non-SSA -> SSA
    # conversion (it folds into renaming) but is treated as pure anyway.
    if isinstance(ins, Copy):
        return [ins.src]
    return []


def _term_operands(term):
    if isinstance(term, Return) and term.value is not None:
        return [term.value]
    if isinstance(term, Branch):
        return [term.cond]
    return []


# --------------------------------------------------------------------------
# Per-function elimination
# --------------------------------------------------------------------------


def _live_definitions(func: Function) -> set:
    """Return the ids of values live from the terminator/call roots.

    Roots are every value read by a ``Return`` or ``Branch`` and *every*
    ``Call`` instruction (plus, transitively, its arguments): a call whose
    result is unused is still observable.  From each root the use-def graph
    -- instruction operands and phi incoming values -- is walked to a
    fixpoint, so a value only reachable from dead code or a closed phi
    cycle is absent from the result.
    """
    phi_by_dest: dict = {}
    operands_by_dest: dict = {}
    worklist: list = []

    for block in func.blocks:
        for ins in block.instructions:
            operands_by_dest[id(ins.dest)] = _operands(ins)
            if isinstance(ins, Call):
                # The call itself is a root even if its result is unused:
                # reaching its dest below also pulls in its arguments.
                worklist.append(ins.dest)
        worklist.extend(_term_operands(block.terminator))
        for phi in block.phis:
            phi_by_dest[id(phi.dest)] = phi

    live: set = set()
    while worklist:
        value = worklist.pop()
        if value is None or id(value) in live:
            continue
        live.add(id(value))
        phi = phi_by_dest.get(id(value))
        if phi is not None:
            worklist.extend(phi.entries.values())
        else:
            worklist.extend(operands_by_dest.get(id(value), ()))
    return live


def _eliminate_in_function(func: Function) -> Function:
    live = _live_definitions(func)

    new_blocks = [Block(block.id) for block in func.blocks]
    block_map = dict(zip(func.blocks, new_blocks))

    # Pre-create one fresh Temp for every surviving definition before
    # cloning anything: a phi in an earlier block (e.g. a loop header's
    # backedge entry) can reference an instruction defined in a later
    # block, and every retained use must resolve to the very same output
    # object.  Original ids are kept, so numbering holes from removed
    # definitions remain.
    value_map: dict = {
        id(param.temp): Temp(param.temp.id, param.temp.type)
        for param in func.params
    }
    for block in func.blocks:
        for phi in block.phis:
            if id(phi.dest) in live:
                value_map[id(phi.dest)] = Temp(phi.dest.id, phi.dest.type)
        for ins in block.instructions:
            if isinstance(ins, Call) or id(ins.dest) in live:
                value_map[id(ins.dest)] = Temp(
                    ins.dest.id, ins.dest.type
                )

    def map_value(value):
        if value is None:
            return None
        mapped = value_map.get(id(value))
        if mapped is None:
            # Defensive: a referenced value with no surviving definition.
            # Never occurs for SSA IR produced by to_ssa, where every
            # operand of a live construct is itself live.
            mapped = Temp(value.id, value.type)
            value_map[id(value)] = mapped
        return mapped

    for block, new_block in zip(func.blocks, new_blocks):
        for phi in block.phis:
            if id(phi.dest) not in live:
                continue
            entries = {
                block_map[pred]: map_value(value)
                for pred, value in sorted(
                    phi.entries.items(), key=lambda item: item[0].id
                )
            }
            new_block.phis.append(Phi(map_value(phi.dest), entries))

        for ins in block.instructions:
            if not isinstance(ins, Call) and id(ins.dest) not in live:
                continue
            dest = map_value(ins.dest)
            if isinstance(ins, Const):
                clone = Const(dest, ins.value)
            elif isinstance(ins, Copy):
                clone = Copy(dest, map_value(ins.src))
            elif isinstance(ins, BinOp):
                clone = BinOp(
                    dest, ins.operator, map_value(ins.left),
                    map_value(ins.right), ins.kind, ins.type,
                )
            else:
                clone = Call(
                    dest, ins.name,
                    [map_value(arg) for arg in ins.args], ins.type,
                )
            new_block.instructions.append(clone)

        term = block.terminator
        if isinstance(term, Return):
            new_block.terminator = Return(map_value(term.value))
        elif isinstance(term, Jump):
            new_block.terminator = Jump(block_map[term.target])
        elif isinstance(term, Branch):
            new_block.terminator = Branch(
                map_value(term.cond),
                block_map[term.true_target],
                block_map[term.false_target],
            )

    new_params = [
        Parameter(
            param.name,
            param.slot,
            map_value(param.temp),
        )
        for param in func.params
    ]

    return Function(
        func.name,
        new_params,
        func.ret_type,
        list(func.locals),
        new_blocks,
        block_map[func.entry],
        ssa=True,
    )
