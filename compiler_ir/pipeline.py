"""Library-level optimization pipeline.

The entry point :func:`optimize_module` collects into one library function
the pass orchestration that callers previously hand-wrote around
:func:`~compiler_ir.ssa.to_ssa`,
:func:`~compiler_ir.folding.fold_constants`,
:func:`~compiler_ir.licm.hoist_loop_invariants` and
:func:`~compiler_ir.dce.eliminate_dead_code`::

    optimized = optimize_module(module)                     # default schedule
    optimized = optimize_module(module, ("ssa", "fold"))   # explicit order
    copied    = optimize_module(module, ())                 # copy only

The accepted inputs are exactly:

* a non-SSA :class:`Module` produced by
  :func:`~compiler_ir.lowerer.lower_module`, or
* an existing SSA :class:`Module` (for example the result of an earlier
  :func:`optimize_module` call).

``passes`` is a finite sequence of the names ``"ssa"``, ``"fold"``,
``"licm"`` and ``"dce"``; passes run in the given order and names may
repeat (``ssa`` freely, and ``fold``/``licm``/``dce`` whenever the SSA
stage constraint holds).  When it is omitted, the schedule is
``("ssa", "fold", "licm", "dce", "ssa")``: SSA construction, constant
folding, loop-invariant code motion, dead-code elimination, and a trailing
SSA canonicalization that renumbers away the definition holes DCE may
leave, so the result is the fixed point of the default sequence.

Stage constraint
----------------

``fold``, ``licm`` and ``dce`` require an SSA module.  On a non-SSA input
the first such name in the sequence must be preceded by an ``ssa``; an SSA
input is already past that stage, so it may start directly with ``fold``,
``licm`` or ``dce`` (an explicit ``ssa`` then merely re-canonicalizes the
numbering).
:func:`~compiler_ir.printer.render_module` is not an optimization and is
never part of a schedule -- it is reported as an unknown name.

An empty sequence performs no optimization: it returns a brand new module
with equivalent content, preserving the input's SSA/non-SSA flavor.  No
schedule, including the empty one, renders text -- the result is always a
traversable :class:`Module` the caller may hand to
:func:`~compiler_ir.printer.render_module` itself.

Every call -- even with an empty schedule, and even on the empty module --
returns fresh function, block, instruction and phi containers; the input
module is never mutated and shares no mutable container with the result.

Validation happens *before* any pass runs.  A non-:class:`Module`
``module`` raises :class:`TypeError`; ``passes`` given as a single string
or any non-sequence, or containing non-string elements, raises
:class:`TypeError`; unknown pass names and ``fold``/``licm``/``dce``
scheduled before the first SSA stage on a non-SSA input raise
:class:`ValueError`.  A rejected call leaves the input module unchanged.
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
from .licm import hoist_loop_invariants
from .ssa import to_ssa

#: Pass names that may appear in an explicit schedule.
PASS_NAMES = ("ssa", "fold", "licm", "dce")

#: Schedule used when ``passes`` is omitted.
DEFAULT_PASSES = ("ssa", "fold", "licm", "dce", "ssa")

_PASS_FUNCS = {
    "ssa": to_ssa,
    "fold": fold_constants,
    "licm": hoist_loop_invariants,
    "dce": eliminate_dead_code,
}

#: Names whose stage precondition is an SSA module.
_SSA_DEPENDENT = frozenset(("fold", "licm", "dce"))


def optimize_module(module: Module, passes=None) -> Module:
    """Run an ordered schedule of optimization passes over ``module``.

    :param module: a non-SSA :class:`Module` from
        :func:`~compiler_ir.lowerer.lower_module` or an existing SSA
        :class:`Module`.
    :param passes: an optional finite sequence of pass names drawn from
        ``"ssa"``, ``"fold"``, ``"licm"`` and ``"dce"``.  When ``None``
        (the default) the schedule is
        ``("ssa", "fold", "licm", "dce", "ssa")``.  An empty sequence
        returns an independent, content-equivalent copy.
    :return: a brand new SSA :class:`Module` for any non-empty schedule;
        for an empty schedule the copy keeps the input's SSA flavor.  The
        result is traversable and renderable, and shares no mutable
        container with the input.
    :raises TypeError: if ``module`` is not a :class:`Module`, or
        ``passes`` is a string, a non-sequence, or contains non-string
        elements.
    :raises ValueError: if ``passes`` contains an unknown name (including
        ``"render_module"``), or schedules ``fold``/``licm``/``dce`` on a
        non-SSA input before any ``ssa`` stage.
    """
    if not isinstance(module, Module):
        raise TypeError(
            "optimize_module expects a Module, got "
            f"{type(module).__name__}"
        )

    if passes is None:
        schedule = DEFAULT_PASSES
    else:
        schedule = _validate_passes(passes)

    if not schedule:
        # No optimization requested: return an independent copy in the
        # same flavor.  An SSA module clones through to_ssa (its documented
        # idempotent copy); a non-SSA module is structurally deep-copied so
        # slots and short-circuit temporaries keep their non-SSA semantics.
        if getattr(module, "ssa", False):
            return to_ssa(module)
        return _clone_non_ssa_module(module)

    # Up-front stage validation over the whole order, before any pass runs.
    # An SSA input starts past the SSA stage; a non-SSA input must meet its
    # first fold/licm/dce only after an ssa entry.
    ssa_stage = bool(getattr(module, "ssa", False))
    for name in schedule:
        if name == "ssa":
            ssa_stage = True
        elif name in _SSA_DEPENDENT and not ssa_stage:
            raise ValueError(
                "pass order violates precondition: "
                f"{name!r} requires an SSA module, but no 'ssa' pass "
                "precedes it"
            )

    current = module
    for name in schedule:
        current = _PASS_FUNCS[name](current)
    return current


def _validate_passes(passes) -> tuple:
    """Validate the user-supplied schedule's shape, independent of module.

    Returns the schedule as a fresh tuple so iteration is stable and
    independent of any caller-owned container.
    """
    # A single string is the classic mistake ("passes='fold'"); reject it
    # before the Sequence membership test would otherwise iterate letters.
    if isinstance(passes, str) or not isinstance(passes, Sequence):
        raise TypeError(
            "optimize_module expects passes to be a sequence of pass name "
            f"strings or None, got {type(passes).__name__}"
        )

    schedule = tuple(passes)
    for index, name in enumerate(schedule):
        if not isinstance(name, str):
            raise TypeError(
                "optimize_module pass names must be strings; element "
                f"{index} is {type(name).__name__}"
            )
        if name not in _PASS_FUNCS:
            # render_module (and "isel", "inline", ...) is intentionally
            # not an optimizable pass: report it like any unknown name.
            raise ValueError(f"unknown pass in order: {name!r}")
    return schedule


# --------------------------------------------------------------------------
# Structural copy of a non-SSA module (empty-schedule path only)
# --------------------------------------------------------------------------


def _clone_non_ssa_module(module: Module) -> Module:
    """Deep-copy a non-SSA :class:`Module` with full container independence.

    Slots and temporaries are immutable value objects and may be shared
    harmlessly, while every function, block, instruction, terminator and
    parameter container is rebuilt and every terminator edge is repointed
    at the cloned block.
    """
    return Module(
        [_clone_non_ssa_function(func) for func in module.functions],
        ssa=False,
    )


def _clone_non_ssa_function(func: Function) -> Function:
    new_blocks = [Block(block.id) for block in func.blocks]
    block_map = dict(zip(func.blocks, new_blocks))

    new_params = [
        Parameter(param.name, param.slot, param.temp)
        for param in func.params
    ]

    for block, new_block in zip(func.blocks, new_blocks):
        # Non-SSA blocks carry no phis; copy defensively so the result has
        # fully independent containers regardless of input provenance.
        for phi in block.phis:
            new_block.phis.append(
                Phi(
                    phi.dest,
                    {
                        block_map[pred]: value
                        for pred, value in phi.entries.items()
                    },
                )
            )

        for ins in block.instructions:
            if isinstance(ins, Const):
                clone = Const(ins.dest, ins.value)
            elif isinstance(ins, Copy):
                clone = Copy(ins.dest, ins.src)
            elif isinstance(ins, BinOp):
                clone = BinOp(
                    ins.dest, ins.operator, ins.left, ins.right,
                    ins.kind, ins.type,
                )
            elif isinstance(ins, Call):
                clone = Call(ins.dest, ins.name, list(ins.args), ins.type)
            else:
                raise AssertionError(  # pragma: no cover
                    f"unknown instruction: {ins!r}")
            new_block.instructions.append(clone)

        term = block.terminator
        if term is None:
            new_block.terminator = None
        elif isinstance(term, Return):
            new_block.terminator = Return(term.value)
        elif isinstance(term, Jump):
            new_block.terminator = Jump(block_map[term.target])
        elif isinstance(term, Branch):
            new_block.terminator = Branch(
                term.cond,
                block_map[term.true_target],
                block_map[term.false_target],
            )
        else:
            raise AssertionError(  # pragma: no cover
                f"unknown terminator: {term!r}")

    return Function(
        func.name,
        new_params,
        func.ret_type,
        list(func.locals),
        new_blocks,
        block_map[func.entry],
        ssa=False,
    )
