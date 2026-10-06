"""Library-level orchestration of the optimization passes.

The entry point :func:`optimize_module` bundles the pass sequence that
previously had to be scheduled by hand into one call::

    optimize_module(module)                      # ssa, fold, dce, ssa
    optimize_module(module, ("ssa", "fold"))     # explicit order
    optimize_module(module, ())                  # independent copy

It accepts either a non-SSA :class:`Module` produced by
:func:`~compiler_ir.lowerer.lower_module` or an existing SSA
:class:`Module`, and returns a brand new module that can be traversed
further and handed to :func:`~compiler_ir.printer.render_module`.  With
``passes`` omitted the default sequence ``ssa, fold, dce, ssa`` runs and
the result is an SSA module at the canonical fixed point of the sequence:
applying the default again to its own result changes neither structure
nor rendered text.

An explicit ``passes`` is a finite sequence of the names ``"ssa"``,
``"fold"`` and ``"dce"``, executed item by item in the given order.
``"ssa"`` may repeat (it re-canonicalizes the numbering of an already-SSA
module); ``"fold"`` and ``"dce"`` may repeat as long as the stage
contract holds -- both require an SSA module, so on a non-SSA input an
``"ssa"`` pass must precede them, while an SSA input may start with
``"fold"`` or ``"dce"`` directly.  An empty sequence returns a copy that
is content-equivalent to the input but shares no mutable container with
it; no text is rendered anywhere in this function.

Validation happens in full before any pass runs:

* a non-:class:`Module` ``module`` raises :class:`TypeError`;
* a ``passes`` that is a string, is not a sequence, or contains
  non-string elements raises :class:`TypeError`;
* an unknown pass name -- rendering is not an optimizable pass, so
  ``"render_module"`` counts as unknown -- or a ``fold``/``dce``
  scheduled before the first ``ssa`` on non-SSA input raises
  :class:`ValueError`.

A failed validation never touches the input, and a successful call never
reuses a mutable function, block, instruction or phi container from the
input -- this holds for the empty module and the empty sequence as well.
Every pass is deterministic, so the same module and the same sequence
always produce structurally and textually identical results.
"""
from collections.abc import Sequence

from .dce import eliminate_dead_code
from .folding import fold_constants
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
)
from .ssa import to_ssa

#: The default optimization pipeline: SSA construction, constant folding,
#: dead-code elimination, and a final SSA re-canonicalization that
#: compacts the numbering holes DCE may leave.
DEFAULT_PASSES = ("ssa", "fold", "dce", "ssa")

_PASS_FUNCS = {
    "ssa": to_ssa,
    "fold": fold_constants,
    "dce": eliminate_dead_code,
}

_SSA_DEPENDENT = ("fold", "dce")


def optimize_module(module: Module, passes=None) -> Module:
    """Run an optimization pass sequence over ``module``.

    ``module`` is a non-SSA :class:`Module` from
    :func:`~compiler_ir.lowerer.lower_module` or an existing SSA
    :class:`Module`; it is never mutated and no mutable container is
    shared with the result.  ``passes`` omitted (or ``None``) selects
    :data:`DEFAULT_PASSES`; an explicit sequence of ``"ssa"``/``"fold"``/
    ``"dce"`` names runs in the given order; an empty sequence returns an
    independent, content-equivalent copy of the input.

    :raises TypeError: if ``module`` is not a :class:`Module`, or
        ``passes`` is a string, not a sequence, or holds non-strings.
    :raises ValueError: for an unknown pass name, or a ``fold``/``dce``
        scheduled before the first ``ssa`` on non-SSA input.
    """
    if not isinstance(module, Module):
        raise TypeError(
            f"optimize_module expects a Module, got {type(module).__name__}"
        )
    names = DEFAULT_PASSES if passes is None else _checked_names(passes)
    _checked_order(module, names)
    if not names:
        return _clone_module(module)
    current = module
    for name in names:
        current = _PASS_FUNCS[name](current)
    return current


# --------------------------------------------------------------------------
# Validation (runs in full before any pass)
# --------------------------------------------------------------------------


def _checked_names(passes) -> tuple:
    """Return ``passes`` as a tuple of strings, or raise :class:`TypeError`."""
    if isinstance(passes, str) or not isinstance(passes, Sequence):
        raise TypeError(
            "optimize_module passes must be a sequence of pass names, got "
            f"{type(passes).__name__}"
        )
    names = tuple(passes)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                "optimize_module pass names must be strings, got "
                f"{type(name).__name__}"
            )
    return names


def _checked_order(module: Module, names: tuple) -> None:
    """Validate every name and the SSA stage contract up front."""
    ssa_seen = bool(module.ssa)
    for name in names:
        if name not in _PASS_FUNCS:
            raise ValueError(f"unknown pass name: {name!r}")
        if name in _SSA_DEPENDENT and not ssa_seen:
            raise ValueError(
                f"pass {name!r} requires an SSA module, but no 'ssa' pass "
                "precedes it"
            )
        if name == "ssa":
            ssa_seen = True


# --------------------------------------------------------------------------
# Flavor-preserving structural clone (used for the empty sequence)
# --------------------------------------------------------------------------


def _clone_module(module: Module) -> Module:
    """Copy ``module`` keeping its SSA/non-SSA flavor and numbering.

    :class:`Temp`/:class:`Slot` references are immutable and may be
    shared; every mutable function, block, instruction, phi, parameter
    and terminator object is fresh.
    """
    return Module(
        [_clone_function(func) for func in module.functions],
        ssa=bool(module.ssa),
    )


def _clone_function(func: Function) -> Function:
    new_blocks = [Block(block.id) for block in func.blocks]
    block_map = dict(zip(func.blocks, new_blocks))

    for block, new_block in zip(func.blocks, new_blocks):
        new_block.phis = [
            Phi(
                phi.dest,
                {
                    block_map[pred]: value
                    for pred, value in sorted(
                        phi.entries.items(), key=lambda item: item[0].id
                    )
                },
            )
            for phi in block.phis
        ]
        new_block.instructions = [
            _clone_instruction(ins) for ins in block.instructions
        ]
        new_block.terminator = _clone_terminator(block.terminator, block_map)

    return Function(
        func.name,
        [Parameter(p.name, p.slot, p.temp) for p in func.params],
        func.ret_type,
        list(func.locals),
        new_blocks,
        block_map[func.entry],
        ssa=func.ssa,
    )


def _clone_instruction(ins):
    if isinstance(ins, Const):
        return Const(ins.dest, ins.value)
    if isinstance(ins, Copy):
        return Copy(ins.dest, ins.src)
    if isinstance(ins, BinOp):
        return BinOp(
            ins.dest, ins.operator, ins.left, ins.right, ins.kind, ins.type
        )
    if isinstance(ins, Call):
        return Call(ins.dest, ins.name, list(ins.args), ins.type)
    raise AssertionError(f"unknown instruction: {ins!r}")  # pragma: no cover


def _clone_terminator(term, block_map):
    if isinstance(term, Return):
        return Return(term.value)
    if isinstance(term, Jump):
        return Jump(block_map[term.target])
    if isinstance(term, Branch):
        return Branch(
            term.cond,
            block_map[term.true_target],
            block_map[term.false_target],
        )
    return None  # pragma: no cover - lowerer always terminates blocks
