"""SSA constant folding and propagation.

The entry point :func:`fold_constants` takes an SSA :class:`Module`
produced by :func:`~compiler_ir.ssa.to_ssa` and returns a brand new SSA
:class:`Module` with every definition whose value is a known literal
rewritten as a ``Const``; the input module is never mutated and shares no
mutable function, block, instruction or phi container with the result.

Known values flow along the SSA definition-use edges:

* a ``Const`` is a known literal;
* an arithmetic (``add``/``sub``/``mul``/``div``/``mod``) or comparison
  (``eq``/``ne``/``lt``/``le``/``gt``/``ge``) ``BinOp`` whose two operands
  are both known folds to a known literal, which then propagates to the
  definitions reading it;
* a ``phi`` folds only when the incoming value of *every* reachable
  predecessor is known and all of them are the same literal; it is then
  rewritten as a ``Const`` of the very same SSA value (same number, same
  type), emitted where the block's ordinary instructions begin.

Integer ``div`` truncates toward zero and ``mod`` satisfies
``a == div(a, b) * b + mod(a, b)``.  A ``div``/``mod`` whose divisor is a
known zero is *not* folded: the original ``BinOp`` is kept verbatim so the
runtime fault, its site and the observable state at the fault are neither
advanced, swallowed nor rewritten.  A ``Call`` result is always unknown;
calls and their relative order are never touched.

The pass rewrites value definitions only.  It deletes no basic block,
turns no constant ``Branch`` into a ``Jump`` and removes no definition
that became unused through folding -- running
:func:`~compiler_ir.dce.eliminate_dead_code` afterwards cleans those up
safely.  Function order, signatures, parameters, block labels and order,
terminators, the relative order of unfolded instructions and the existing
SSA numbers are all preserved, so applying the pass to its own output is a
structural and textual fixed point.  The empty module and modules with
nothing to fold return independent, equivalent copies.  A
non-:class:`Module` object raises :class:`TypeError`; a non-SSA
:class:`Module` raises :class:`ValueError`.
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


def fold_constants(module: Module) -> Module:
    """Return a new SSA :class:`Module` with known literals folded.

    The input module is left untouched; the result shares no mutable
    function, block, instruction or phi object with it.

    :raises TypeError: if ``module`` is not a :class:`Module` instance.
    :raises ValueError: if ``module`` is not in SSA form.
    """
    if not isinstance(module, Module):
        raise TypeError(
            "fold_constants expects a Module, got "
            f"{type(module).__name__}"
        )
    if not getattr(module, "ssa", False):
        raise ValueError("fold_constants expects an SSA Module")
    return Module(
        [_fold_in_function(func) for func in module.functions], ssa=True
    )


# --------------------------------------------------------------------------
# Constant evaluation
# --------------------------------------------------------------------------


# Sentinel for "not a known literal"; distinct from every int/bool value.
_UNKNOWN = object()


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncated toward zero (``b`` is never zero here)."""
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient


def _trunc_mod(a: int, b: int) -> int:
    """Remainder matching truncated division: ``a == div(a, b) * b + mod``."""
    return a - _trunc_div(a, b) * b


_ARITH = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _trunc_div,
    "mod": _trunc_mod,
}

_COMPARE = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def _evaluate(ins: BinOp, left, right):
    """Fold a BinOp on known operands; ``None`` means "do not fold".

    A known zero divisor keeps the ``div``/``mod`` BinOp unfoldable: the
    runtime fault must fire at its original site, not at compile time.
    """
    if ins.kind == "arith":
        if ins.operator in ("div", "mod") and right == 0:
            return None
        operation = _ARITH.get(ins.operator)
    else:
        operation = _COMPARE.get(ins.operator)
    if operation is None:
        return None
    return operation(left, right)


def _same_literal(a, b) -> bool:
    """Literal equality that keeps ``int`` and ``bool`` apart."""
    return type(a) is type(b) and a == b


# --------------------------------------------------------------------------
# Known-value propagation (fixpoint over the definition-use graph)
# --------------------------------------------------------------------------


def _known_values(func: Function) -> dict:
    """Map ``id(defining temp)`` to its literal for every known definition.

    Parameters and ``Call`` results are never known.  Facts only
    accumulate, so iterating the definitions in their fixed module order to
    a fixpoint is deterministic; the loop is needed because a loop-header
    phi can reference a value defined in a later block on the back edge.
    """
    binops: dict = {}
    copies: dict = {}
    phis: dict = {}
    known: dict = {}
    for block in func.blocks:
        for ins in block.instructions:
            if isinstance(ins, Const):
                known[id(ins.dest)] = ins.value
            elif isinstance(ins, BinOp):
                binops[id(ins.dest)] = ins
            elif isinstance(ins, Copy):
                # A Copy cannot survive to_ssa, but it is pure: propagate.
                copies[id(ins.dest)] = ins
        for phi in block.phis:
            phis[id(phi.dest)] = phi

    changed = True
    while changed:
        changed = False
        for dest_id, ins in binops.items():
            if dest_id in known:
                continue
            left = known.get(id(ins.left), _UNKNOWN)
            right = known.get(id(ins.right), _UNKNOWN)
            if left is _UNKNOWN or right is _UNKNOWN:
                continue
            value = _evaluate(ins, left, right)
            if value is not None:
                known[dest_id] = value
                changed = True
        for dest_id, ins in copies.items():
            if dest_id in known:
                continue
            src = known.get(id(ins.src), _UNKNOWN)
            if src is not _UNKNOWN:
                known[dest_id] = src
                changed = True
        for dest_id, phi in phis.items():
            if dest_id in known:
                continue
            incoming = []
            for value in phi.entries.values():
                literal = (
                    _UNKNOWN if value is None
                    else known.get(id(value), _UNKNOWN)
                )
                if literal is _UNKNOWN:
                    break
                incoming.append(literal)
            else:
                if incoming and all(
                    _same_literal(incoming[0], other)
                    for other in incoming[1:]
                ):
                    known[dest_id] = incoming[0]
                    changed = True
    return known


# --------------------------------------------------------------------------
# Per-function rewrite
# --------------------------------------------------------------------------


def _fold_in_function(func: Function) -> Function:
    known = _known_values(func)

    new_blocks = [Block(block.id) for block in func.blocks]
    block_map = dict(zip(func.blocks, new_blocks))

    # Pre-create one fresh Temp per definition before cloning anything, so
    # a use that precedes its definition in block order (a loop-header
    # phi's backedge entry) resolves to the very same output object.
    # Original SSA ids are kept: folding never renumbers.
    value_map: dict = {
        id(param.temp): Temp(param.temp.id, param.temp.type)
        for param in func.params
    }
    for block in func.blocks:
        for phi in block.phis:
            value_map[id(phi.dest)] = Temp(phi.dest.id, phi.dest.type)
        for ins in block.instructions:
            value_map[id(ins.dest)] = Temp(ins.dest.id, ins.dest.type)

    def map_value(value):
        if value is None:
            return None
        mapped = value_map.get(id(value))
        if mapped is None:
            # Defensive: a referenced value with no definition in the
            # function.  Never occurs for SSA IR produced by to_ssa.
            mapped = Temp(value.id, value.type)
            value_map[id(value)] = mapped
        return mapped

    for block, new_block in zip(func.blocks, new_blocks):
        for phi in block.phis:
            folded = known.get(id(phi.dest), _UNKNOWN)
            if folded is not _UNKNOWN:
                # A phi whose reachable inputs are all the same literal
                # becomes a Const of the same SSA value; it opens the
                # block's ordinary instructions, where phis used to sit.
                new_block.instructions.append(
                    Const(map_value(phi.dest), folded)
                )
                continue
            entries = {
                block_map[pred]: map_value(value)
                for pred, value in sorted(
                    phi.entries.items(), key=lambda item: item[0].id
                )
            }
            new_block.phis.append(Phi(map_value(phi.dest), entries))

        for ins in block.instructions:
            dest = map_value(ins.dest)
            folded = known.get(id(ins.dest), _UNKNOWN)
            if isinstance(ins, Const):
                clone = Const(dest, ins.value)
            elif isinstance(ins, Copy):
                clone = Copy(dest, map_value(ins.src))
            elif isinstance(ins, BinOp):
                if folded is not _UNKNOWN:
                    clone = Const(dest, folded)
                else:
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
        Parameter(param.name, param.slot, map_value(param.temp))
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
