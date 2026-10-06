"""Pass-ordering and idempotence semantic regression tests.

The existing suites pin each pass in isolation (lowering, non-SSA -> SSA,
SSA dead-code elimination) and compare non-SSA vs SSA behavior.  This module
instead treats the compiler as a *pipeline* made of the existing public
entry points and checks that:

1. One legal source program, fed through the existing entry point
   (:func:`lower_module`, which still runs AST validation, name resolution
   and type checking), produces the same observable behavior under the
   default optimization pipeline and every alternative *legal* pass order,
   using the unoptimized (non-SSA) result as the semantic baseline.
2. For fixed source, options and order, repeated compilations are
   byte-identical (determinism), while *different* orders are only required
   to agree on observable behavior -- never on instruction or text layout.
3. Re-applying the same pass sequence to an optimized result changes
   neither the canonicalized IR nor the emitted target text (idempotence);
   each repeatable pass is also a fixed point on its own.
4. Orderings may only permute passes inside the existing stage contracts:
   SSA construction precedes every SSA-dependent optimization, and
   instruction selection (here, the deterministic :func:`render_module`
   emission) runs after IR optimization.

Passes and preconditions (nothing new is added to the compiler package)
-----------------------------------------------------------------------

* ``ssa``  -- :func:`to_ssa`.  Legal on a lowered module and, as an
  idempotent canonicalization (clone + deterministic renumbering), on an
  already-SSA module.
* ``dce``  -- :func:`eliminate_dead_code`.  Its documented precondition is
  an SSA module: applying it before ``ssa`` raises ``ValueError``.
* instruction selection is the final :func:`render_module` and is fixed
  last.

The default optimization pipeline is ``ssa, dce, ssa``: the SSA-dependent
optimization (DCE) sits after SSA construction, and a trailing SSA
canonicalization compacts the numbering holes DCE may leave so the emitted
IR is the canonical fixed point of the sequence.  Alternative legal orders
differ only in the middle (extra normalization, an extra fixpoint DCE, or an
ssa/dce interleaving); the no-DCE order ``ssa`` is legal as well.

Observables
-----------

The language has no I/O statement, so the ordered trace of executed calls
``(callee, positional args)`` is the standard-output surrogate -- the only
side effect the language can have (it is also the observable used by
``test_semantic_equivalence.py``).  An outcome is one of:

* normal termination: return value (``None`` for ``void``) + call trace;
* a runtime fault: error category, trigger site, the operands at the fault,
  and the call trace produced *before* the fault.

The language's grammar already includes ``div``/``mod``; the single runtime
error the language defines is a zero divisor.  Division and modulo follow
truncation toward zero (the remainder satisfies ``a == (a/b)*b + a%b``).
There is no array or indexing construct, so no bounds error exists in this
language subset, and no undefined-behavior sample is used: every faulting
instruction's result is consumed (returned, branched on or accumulated), so
no legal order can legally delete it.

A fault site is identified independently of SSA numbering by
``(function name, block label, operator, ordinal of the BinOp among the
BinOps of its block)``: block labels are assigned by the lowering DFS and
preserved by ``to_ssa`` and DCE, and the relative order of surviving
BinOps is preserved as well.  Every faulting block in this suite contains a
single faulting BinOp, so the site is unambiguous.

All inputs are fixed literal argument tuples; the pipeline has no random
or environment-dependent behavior, hence there is no seed to pin.
"""
import copy
import unittest

from compiler_ir import (
    BinOp,
    Branch,
    Call,
    Const,
    Copy,
    Jump,
    Module,
    Return,
    Slot,
    Temp,
    TypeCheckError,
    UndefinedSymbolError,
    eliminate_dead_code,
    lower_module,
    render_module,
    to_ssa,
)

from test_pipeline import (
    arith,
    assign,
    bool_,
    call,
    compare,
    func,
    if_,
    int_,
    let,
    param,
    program,
    ret,
    var,
    while_,
)


# ==========================================================================
# IR interpreter with ordered call output and defined runtime faults
# ==========================================================================


_STEP_LIMIT = 100_000


class RuntimeFault(Exception):
    """A runtime error defined by the language.

    Carries the error ``category``, the numbering-independent ``site``
    ``(function, block label, operator, binop ordinal)``, the operand
    values at the fault, and the call trace observed before it fired.
    """

    def __init__(self, category, function, block_label, operator, ordinal,
                 left, right, output):
        self.category = category
        self.function = function
        self.block_label = block_label
        self.operator = operator
        self.ordinal = ordinal
        self.left = left
        self.right = right
        self.site = (function, block_label, operator, ordinal)
        self.output = list(output)
        super().__init__(
            f"{category} at {function}:{block_label} {operator} "
            f"({left!r}, {right!r})"
        )


ZERO_DIVISOR = "division-by-zero"


def _trunc_div(a: int, b: int):
    """Integer division truncated toward zero; raises ZeroDivisionError."""
    if b == 0:
        raise ZeroDivisionError
    quotient = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        quotient = -quotient
    return quotient


def _trunc_mod(a: int, b: int):
    """Truncated-toward-zero remainder; raises ZeroDivisionError."""
    if b == 0:
        raise ZeroDivisionError
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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _Interpreter:
    """Execute a lowered :class:`Module` of either SSA flavor.

    Mirrors the interpreter in ``test_semantic_equivalence.py`` but adds
    ``div``/``mod`` and converts their zero-divisor trap into a
    :class:`RuntimeFault` with a numbering-independent site.
    """

    def __init__(self, module: Module):
        _assert(isinstance(module, Module),
                f"interpreter expects a Module, got {type(module).__name__}")
        self.ssa = bool(module.ssa)
        self._functions = {fn_.name: fn_ for fn_ in module.functions}

    def run(self, entry: str, arguments):
        fn_ = self._functions.get(entry)
        _assert(fn_ is not None, f"unknown entry function {entry!r}")
        arguments = tuple(arguments)
        _assert(
            len(arguments) == len(fn_.params),
            f"entry {entry!r} expects {len(fn_.params)} argument(s), "
            f"got {len(arguments)}",
        )
        trace: list = []
        value = self._invoke(fn_, arguments, trace)
        return value, trace

    def _invoke(self, fn_, arguments, trace: list):
        definitions: dict = {}
        slots: dict = {}
        temps: dict = {}

        for parameter, argument in zip(fn_.params, arguments):
            if self.ssa:
                _assert(parameter.temp is not None,
                        f"SSA parameter {parameter.name!r} has no definition")
                definitions[id(parameter.temp)] = argument
            else:
                slots[parameter.slot.id] = argument

        def write_ref(ref, value) -> None:
            if self.ssa:
                _assert(isinstance(ref, Temp) and not isinstance(
                    ref, Slot),
                    f"{fn_.name}: non-Temp definition {ref!r} in SSA IR")
                definitions[id(ref)] = value
            elif isinstance(ref, Slot):
                slots[ref.id] = value
            else:
                temps[id(ref)] = value

        def read(ref):
            if isinstance(ref, Slot):
                _assert(ref.id in slots,
                        f"{fn_.name}: read of undefined slot {ref}")
                return slots[ref.id]
            if self.ssa:
                _assert(isinstance(ref, Temp),
                        f"{fn_.name}: SSA read of non-Temp value {ref!r}")
                _assert(id(ref) in definitions,
                        f"{fn_.name}: read of undefined SSA value {ref}")
                return definitions[id(ref)]
            _assert(id(ref) in temps,
                    f"{fn_.name}: read of undefined temporary {ref}")
            return temps[id(ref)]

        def execute(instruction, block, binop_index) -> None:
            if isinstance(instruction, Const):
                write_ref(instruction.dest, instruction.value)
            elif isinstance(instruction, Copy):
                write_ref(instruction.dest, read(instruction.src))
            elif isinstance(instruction, BinOp):
                left = read(instruction.left)
                right = read(instruction.right)
                try:
                    value = _apply_binop(instruction, left, right, fn_.name)
                except ZeroDivisionError:
                    raise RuntimeFault(
                        ZERO_DIVISOR, fn_.name, block.label,
                        instruction.operator, binop_index[id(instruction)],
                        left, right, trace,
                    )
                write_ref(instruction.dest, value)
            elif isinstance(instruction, Call):
                callee = self._functions.get(instruction.name)
                _assert(callee is not None,
                        f"{fn_.name}: call to unknown function "
                        f"{instruction.name!r}")
                values = tuple(read(operand) for operand in instruction.args)
                _assert(
                    len(values) == len(callee.params),
                    f"call to {instruction.name!r}: run-time arity mismatch",
                )
                trace.append((instruction.name, values))
                write_ref(instruction.dest,
                          self._invoke(callee, values, trace))
            else:
                _assert(False,
                        f"{fn_.name}: unknown instruction {instruction!r}")

        known_blocks = {id(block) for block in fn_.blocks}
        block = fn_.entry
        previous = None
        steps_left = _STEP_LIMIT

        while True:
            _assert(steps_left > 0,
                    f"{fn_.name}: step budget exhausted "
                    "(program does not terminate?)")
            steps_left -= 1

            if self.ssa:
                pending = []
                for phi in block.phis:
                    _assert(previous is not None,
                            f"{fn_.name}:{block.label}: phi in entry block "
                            "has no predecessor edge")
                    source = phi.entries.get(previous)
                    _assert(source is not None,
                            f"{fn_.name}:{block.label}: phi {phi.dest} has "
                            f"no incoming value from {previous.label}")
                    pending.append((phi.dest, read(source)))
                for destination, value in pending:
                    definitions[id(destination)] = value
            else:
                _assert(not block.phis,
                        f"{fn_.name}:{block.label}: phi node in non-SSA IR")

            # Numbering-independent ordinal of each BinOp among the block's
            # BinOps (DCE preserves their relative order).  BinOp nodes are
            # mutable dataclasses and therefore unhashable, so key by id.
            binop_index = {
                id(ins): ordinal
                for ordinal, ins in enumerate(
                    ins for ins in block.instructions
                    if isinstance(ins, BinOp)
                )
            }
            for instruction in block.instructions:
                execute(instruction, block, binop_index)

            terminator = block.terminator
            _assert(terminator is not None,
                    f"{fn_.name}:{block.label}: missing terminator")

            if isinstance(terminator, Return):
                return None if terminator.value is None else read(
                    terminator.value)

            if isinstance(terminator, Jump):
                _assert(id(terminator.target) in known_blocks,
                        f"{fn_.name}:{block.label}: jump to unknown block "
                        f"{terminator.target.label}")
                previous, block = block, terminator.target
                continue

            if isinstance(terminator, Branch):
                condition = read(terminator.cond)
                _assert(isinstance(condition, bool),
                        f"{fn_.name}:{block.label}: branch on non-bool "
                        f"value {condition!r}")
                target = (terminator.true_target if condition
                          else terminator.false_target)
                _assert(id(target) in known_blocks,
                        f"{fn_.name}:{block.label}: branch to unknown block "
                        f"{target.label}")
                previous, block = block, target
                continue

            _assert(False,
                    f"{fn_.name}:{block.label}: unknown terminator")


def _apply_binop(instruction: BinOp, left, right, fn_name: str):
    if instruction.kind == "arith":
        _assert(type(left) is int and type(right) is int,
                f"{fn_name}: arithmetic {instruction.operator!r} on "
                f"non-int operands {left!r}, {right!r}")
        operation = _ARITH.get(instruction.operator)
        _assert(operation is not None,
                f"{fn_name}: unsupported arithmetic operator "
                f"{instruction.operator!r}")
        return operation(left, right)

    if instruction.kind == "compare":
        _assert(type(left) == type(right),
                f"{fn_name}: comparison {instruction.operator!r} between "
                f"mismatched value types {left!r}, {right!r}")
        operation = _COMPARE.get(instruction.operator)
        _assert(operation is not None,
                f"{fn_name}: unsupported comparison operator "
                f"{instruction.operator!r}")
        return operation(left, right)

    _assert(False, f"{fn_name}: unknown BinOp kind {instruction.kind!r}")


class Outcome:
    """Observable result of one run: either a normal value or a fault."""

    def __init__(self, kind, value=None, output=(), category=None,
                 site=None, operands=None):
        self.kind = kind  # "normal" | "fault"
        self.value = value
        self.output = list(output)
        self.category = category
        self.site = site
        self.operands = operands

    @classmethod
    def normal(cls, value, output):
        return cls("normal", value=value, output=output)

    @classmethod
    def fault(cls, error: RuntimeFault):
        return cls(
            "fault", output=error.output, category=error.category,
            site=error.site, operands=(error.left, error.right),
        )

    def __eq__(self, other):
        if not isinstance(other, Outcome):
            return NotImplemented
        return (
            self.kind, self.value, self.output,
            self.category, self.site, self.operands,
        ) == (
            other.kind, other.value, other.output,
            other.category, other.site, other.operands,
        )

    def __repr__(self):
        if self.kind == "normal":
            return f"Outcome.normal(value={self.value!r}, output={self.output!r})"
        return (
            f"Outcome.fault(category={self.category!r}, site={self.site!r}, "
            f"operands={self.operands!r}, output={self.output!r})"
        )


def _interpret(module: Module, entry: str, arguments) -> Outcome:
    try:
        value, trace = _Interpreter(module).run(entry, arguments)
    except RuntimeFault as fault:
        return Outcome.fault(fault)
    return Outcome.normal(value, trace)


# ==========================================================================
# Pass schedules (existing passes only; stage contracts enforced)
# ==========================================================================


# SSA construction, DCE, then a final SSA canonicalization (clone +
# deterministic renumbering), which makes the emitted IR the fixed point of
# the whole sequence: DCE may leave numbering holes, and the trailing ssa
# pass compacts them without changing structure or semantics.
DEFAULT_ORDER = ("ssa", "dce", "ssa")

# Legal alternatives: extra early normalization, an explicit DCE fixpoint
# repeat, and an ssa/dce interleaving -- all end in the canonicalizing ssa
# pass, so each is a fixed point of its own sequence.
ALT_DOUBLE_SSA = ("ssa", "ssa", "dce", "ssa")
ALT_DCE_FIXPOINT = ("ssa", "dce", "dce", "ssa")
ALT_INTERLEAVE = ("ssa", "dce", "ssa", "dce", "ssa")
# Legal but less optimizing: construction + canonicalization only.
NO_DCE_ORDER = ("ssa",)
# The README-documented core without the trailing canonicalization: legal,
# used for cross-order semantics and determinism (its endpoint may carry
# numbering holes and is compared on observables, not text).
RAW_CORE_ORDER = ("ssa", "dce")

LEGAL_ORDERS = {
    "default ssa,dce,ssa": DEFAULT_ORDER,
    "alt ssa,ssa,dce,ssa": ALT_DOUBLE_SSA,
    "alt ssa,dce,dce,ssa": ALT_DCE_FIXPOINT,
    "alt ssa,dce,ssa,dce,ssa": ALT_INTERLEAVE,
    "no-dce ssa": NO_DCE_ORDER,
    "raw core ssa,dce": RAW_CORE_ORDER,
}

# Orders whose endpoint is the canonical fixed point used by the strict
# idempotence test.
IDEMPOTENT_ORDERS = {
    name: order for name, order in LEGAL_ORDERS.items()
    if name != "raw core ssa,dce"
}


def apply_order(module: Module, order) -> Module:
    """Apply ``order`` (a tuple of pass names) to ``module``.

    :raises ValueError: if a pass name is unknown, if instruction selection
        appears anywhere but the end, or if ``dce`` is scheduled before the
        first ``ssa`` (DCE requires an SSA module).
    """
    ssa_seen = bool(getattr(module, "ssa", False))
    for index, name in enumerate(order):
        if name == "ssa":
            module = to_ssa(module)
            ssa_seen = True
        elif name == "dce":
            if not ssa_seen:
                raise ValueError(
                    "pass order violates precondition: 'dce' requires an SSA "
                    "module, but no 'ssa' pass precedes it"
                )
            module = eliminate_dead_code(module)
        elif name == "isel":
            raise ValueError(
                "pass order violates stage contract: instruction selection "
                "('isel') must be the final stage"
            )
        else:
            raise ValueError(f"unknown pass in order: {name!r}")
    return module


def compile_ast(ast: dict, order):
    """Compile a fresh copy of ``ast`` through lowering and ``order``.

    Returns ``(unoptimized_module, optimized_module, target_text)`` where
    the target text is instruction selection (rendering) run last.
    """
    lowered = lower_module(copy.deepcopy(ast))
    optimized = apply_order(lowered, order)
    return lowered, optimized, render_module(optimized)


def structural_signature(module: Module):
    """Numbering-independent structural fingerprint of a module."""
    signature = []
    for fn_ in module.functions:
        blocks = []
        for block in fn_.blocks:
            instructions = []
            for ins in block.instructions:
                if isinstance(ins, Const):
                    instructions.append(("const", ins.value))
                elif isinstance(ins, Copy):
                    instructions.append(("copy",))
                elif isinstance(ins, BinOp):
                    instructions.append((ins.kind, ins.operator))
                elif isinstance(ins, Call):
                    instructions.append(("call", ins.name, len(ins.args)))
            terminator = type(block.terminator).__name__
            blocks.append((
                block.id, terminator, len(block.phis),
                tuple(instructions),
            ))
        signature.append((
            fn_.name, fn_.ret_type,
            tuple((p.name, p.temp.type if p.temp is not None
                   else p.slot.type) for p in fn_.params),
            tuple(blocks),
        ))
    return tuple(signature)


# ==========================================================================
# Test programs (public AST subset only; helpers double as observable output)
# ==========================================================================


def _emit_function(name="emit"):
    return func(name, [param("v", "int")], "int", [ret(var("v"))])


# A) Constants and phi propagation at a branch convergence.
#
# `a` is the same dominating constant on BOTH outgoing edges, so its merge
# phi is trivial (one incoming SSA value) and is pruned during SSA
# construction; the returned value is that constant propagated through the
# join.  `b` carries genuinely different values per edge and needs a real
# merge phi consumed by a side-effecting call.
def _join_program():
    return program(
        func("join", [param("c", "bool"), param("x", "int")], "int", [
            let("k", "int", int_(7)),
            let("a", "int", int_(0)),
            if_(var("c"), [assign("a", var("k"))],
                 [assign("a", var("k"))]),
            let("b", "int", int_(0)),
            if_(var("c"),
                [assign("b", arith("add", var("x"), int_(1)))],
                [assign("b", arith("sub", var("x"), int_(1)))]),
            let("s", "int", call("emit", [var("b")])),
            ret(arith("add", var("a"), var("s"))),
        ]),
        _emit_function(),
    )


# B) Deletable pure computations vs. still-side-effecting expressions.
#
# `d` and `junk` are pure chains nobody consumes: DCE deletes them (and
# their constants), but BOTH calls remain roots even though `ignored`'s
# result is never read -- deleting either would drop observable output.
def _clean_program():
    return program(
        func("clean", [param("n", "int")], "int", [
            let("d", "int",
                arith("mul", arith("add", int_(1), int_(2)), int_(3))),
            let("keptarg", "int", arith("add", var("n"), int_(10))),
            let("v", "int", call("observe", [var("keptarg")])),
            let("ignored", "int", call("observe", [int_(99)])),
            let("junk", "int",
                arith("add", arith("mul", var("v"), int_(3)), var("d"))),
            ret(arith("add", var("v"), int_(4))),
        ]),
        _emit_function("observe"),
    )


# C) Loop-invariant value plus a conditional branch inside the loop.
#
# `base` is loop invariant and defined before the header; the loop still
# carries real header phis for `i` and `total`, and the in-loop if makes the
# side-effecting call happen only on iterations that take its else edge.
def _loop_program():
    return program(
        func("loopsum", [param("n", "int"), param("k", "int")], "int", [
            let("base", "int", arith("add", var("k"), int_(1))),
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                if_(compare("gt", var("i"), int_(1)),
                    [assign("total", arith(
                        "add",
                        arith("add", var("total"), var("base")),
                        var("i")))],
                    [
                        let("t", "int", call("ping", [var("i")])),
                        assign("total", arith("add", var("total"), var("t"))),
                    ]),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            ret(var("total")),
        ]),
        _emit_function("ping"),
    )


# D) An inlineable call with dead code introduced after it.
#
# `dbl` is a trivial inline candidate; no pass order actually inlines it, so
# every order keeps the call and its observable position.  The pure
# computations built AFTER the call (`junk1`, `junk2`) are dead and vanish
# under DCE, while the later side-effecting call `after` is preserved.
def _around_program():
    return program(
        func("around", [param("n", "int")], "int", [
            let("v", "int", call("dbl", [arith("add", var("n"), int_(1))])),
            let("junk1", "int", arith("mul", var("v"), int_(9))),
            let("junk2", "int", arith("add", var("junk1"), int_(5))),
            let("w", "int", call("after", [var("v")])),
            ret(var("v")),
        ]),
        func("dbl", [param("x", "int")], "int",
             [ret(arith("mul", var("x"), int_(2)))]),
        _emit_function("after"),
    )


# E1) Zero divisor on one arm of a branch; the other arm terminates fine.
def _guarded_fault_program():
    return program(
        func("guarded", [param("c", "bool"), param("x", "int")], "int", [
            let("d", "int", int_(0)),
            if_(var("c"),
                [
                    let("e", "int", call("emit", [int_(1)])),
                    let("q", "int", arith("div", var("x"), var("d"))),
                    ret(arith("add", var("q"), var("e"))),
                ],
                [
                    let("e2", "int", call("emit", [int_(2)])),
                    ret(var("e2")),
                ]),
        ]),
        _emit_function(),
    )


# E2) Zero divisor reached inside a loop after observable output.
def _loop_fault_program():
    return program(
        func("looptrap", [param("n", "int")], "int", [
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("e", "int", call("emit", [var("i")])),
                let("q", "int",
                    arith("div", int_(100),
                          arith("sub", int_(2), var("i")))),
                assign("total", arith("add", var("total"), var("q"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            ret(var("total")),
        ]),
        _emit_function(),
    )


# E3) Modulo by zero after a side effect; single-block fault site.
def _mod_fault_program():
    return program(
        func("modfall", [param("x", "int")], "int", [
            let("z", "int", int_(0)),
            let("e", "int", call("emit", [var("x")])),
            let("r", "int", arith("mod", var("x"), var("z"))),
            ret(arith("add", var("r"), var("e"))),
        ]),
        _emit_function(),
    )


# (label, AST builder, entry, argument tuple, pinned expected Outcome) -- the
# pin anchors ground truth independently of the non-SSA/SSA cross-check.
_CASES = [
    (
        "branch-join constants and phi propagation",
        _join_program, "join", (True, 10),
        Outcome.normal(18, [("emit", (11,))]),
    ),
    (
        "branch-join constants and phi propagation (other edge)",
        _join_program, "join", (False, 10),
        Outcome.normal(16, [("emit", (9,))]),
    ),
    (
        "deletable computations vs side effects",
        _clean_program, "clean", (5,),
        Outcome.normal(19, [("observe", (15,)), ("observe", (99,))]),
    ),
    (
        "deletable computations vs side effects (negative arg)",
        _clean_program, "clean", (-3,),
        Outcome.normal(11, [("observe", (7,)), ("observe", (99,))]),
    ),
    (
        "loop invariant with inner branch",
        _loop_program, "loopsum", (4, 100),
        Outcome.normal(
            208, [("ping", (0,)), ("ping", (1,))]),
    ),
    (
        "loop zero trips",
        _loop_program, "loopsum", (0, 100),
        Outcome.normal(0, []),
    ),
    (
        "inlineable call followed by dead code",
        _around_program, "around", (4,),
        Outcome.normal(10, [("dbl", (5,)), ("after", (10,))]),
    ),
    (
        "branch-arm division by zero",
        _guarded_fault_program, "guarded", (True, 7),
        Outcome(
            "fault",
            output=[("emit", (1,))],
            category=ZERO_DIVISOR,
            site=("guarded", "b1", "div", 0),
            operands=(7, 0),
        ),
    ),
    (
        "branch other arm terminates normally",
        _guarded_fault_program, "guarded", (False, 7),
        Outcome.normal(2, [("emit", (2,))]),
    ),
    (
        "loop division by zero after output",
        _loop_fault_program, "looptrap", (5,),
        Outcome(
            "fault",
            output=[("emit", (0,)), ("emit", (1,)), ("emit", (2,))],
            category=ZERO_DIVISOR,
            # b2 computes the divisor (2 - i) first (BinOp ordinal 0) and
            # then the division (ordinal 1) that actually faults.
            site=("looptrap", "b2", "div", 1),
            operands=(100, 0),
        ),
    ),
    (
        "loop exits just before the faulting iteration",
        _loop_fault_program, "looptrap", (2,),
        Outcome.normal(150, [("emit", (0,)), ("emit", (1,))]),
    ),
    (
        "modulo by zero",
        _mod_fault_program, "modfall", (5,),
        Outcome(
            "fault",
            output=[("emit", (5,))],
            category=ZERO_DIVISOR,
            site=("modfall", "b0", "mod", 0),
            operands=(5, 0),
        ),
    ),
]


# ==========================================================================
# Failure formatting: sample, order, and first differing observable
# ==========================================================================


def _first_difference(expected: Outcome, actual: Outcome):
    """Name the first observable that differs, in comparison order."""
    if expected.kind != actual.kind:
        return "termination kind", expected.kind, actual.kind
    if expected.output != actual.output:
        return "standard output (ordered call trace)", expected.output, \
            actual.output
    if expected.kind == "fault":
        if expected.category != actual.category:
            return "runtime error category", expected.category, \
                actual.category
        if expected.site != actual.site:
            return "runtime error trigger site", expected.site, actual.site
        if expected.operands != actual.operands:
            return "runtime error operands", expected.operands, \
                actual.operands
    else:
        if expected.value != actual.value:
            return "return value", expected.value, actual.value
    return None


def _mismatch_message(label, order_name, entry, arguments, expected, actual):
    diff = _first_difference(expected, actual)
    header = (
        f"semantic mismatch for sample {label!r}, order {order_name!r}, "
        f"entry {entry!r}, arguments {arguments!r}"
    )
    if diff is None:
        return header
    field, wanted, got = diff
    return (
        f"{header}\n"
        f"first differing observable: {field}\n"
        f"  expected: {wanted!r}\n"
        f"  actual:   {got!r}\n"
        f"baseline outcome: {expected!r}\n"
        f"this order:       {actual!r}"
    )


# ==========================================================================
# Tests
# ==========================================================================


class CrossOrderSemanticsTests(unittest.TestCase):
    def test_every_legal_order_matches_unoptimized_baseline(self):
        for label, builder, entry, arguments, pinned in _CASES:
            ast = builder()
            with self.subTest(sample=label):
                baseline = _interpret(
                    lower_module(copy.deepcopy(ast)), entry, arguments)

                # The pin is independent ground truth; baseline must meet it.
                diff = _first_difference(pinned, baseline)
                self.assertIsNone(
                    diff,
                    msg=(f"sample {label!r}: unoptimized baseline disagrees "
                         f"with pinned expectation: {diff}\n"
                         f"baseline: {baseline!r}\n"
                         f"pinned:   {pinned!r}"),
                )

                for order_name, order in LEGAL_ORDERS.items():
                    with self.subTest(order=order_name):
                        _l, optimized, _t = compile_ast(ast, order)
                        outcome = _interpret(optimized, entry, arguments)
                        diff = _first_difference(baseline, outcome)
                        self.assertIsNone(
                            diff,
                            msg=_mismatch_message(
                                label, order_name, entry, arguments,
                                baseline, outcome),
                        )

    def test_target_text_is_never_the_cross_order_correctness_basis(self):
        # The dead-computation sample lays out differently with and without
        # DCE: that text difference is allowed.  Semantics still agree.
        ast = _clean_program()
        _l, optimized, opt_text = compile_ast(ast, DEFAULT_ORDER)
        _l2, unoptimized_ssa, no_dce_text = compile_ast(ast, NO_DCE_ORDER)

        # The pure `(1+2)*3` chain is present without DCE but gone with it.
        # Match a standalone literal 1 ("const 10" must not count).
        const_one = r"const 1(?!\d)"
        self.assertRegex(no_dce_text, const_one)
        self.assertNotRegex(opt_text, const_one)
        self.assertNotEqual(no_dce_text, opt_text)

        for arguments in ((5,), (-3,)):
            a = _interpret(unoptimized_ssa, "clean", arguments)
            b = _interpret(optimized, "clean", arguments)
            self.assertEqual(
                a, b,
                msg="text layouts differ by design; outcomes must not",
            )

    def test_fault_category_site_and_prefix_agree_across_orders(self):
        fault_cases = [
            case for case in _CASES if case[4].kind == "fault"
        ]
        self.assertTrue(fault_cases, "suite must include runtime-fault cases")
        for label, builder, entry, arguments, pinned in fault_cases:
            ast = builder()
            baseline = _interpret(
                lower_module(copy.deepcopy(ast)), entry, arguments)
            self.assertEqual(baseline.kind, "fault")
            self.assertEqual(baseline.category, pinned.category)
            self.assertEqual(baseline.site, pinned.site)
            for order_name, order in LEGAL_ORDERS.items():
                _l, optimized, _t = compile_ast(ast, order)
                outcome = _interpret(optimized, entry, arguments)
                diff = _first_difference(baseline, outcome)
                self.assertIsNone(
                    diff,
                    msg=_mismatch_message(
                        label, order_name, entry, arguments,
                        baseline, outcome),
                )


class DeterminismTests(unittest.TestCase):
    def test_same_order_recompiles_byte_identically(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            for order_name, order in LEGAL_ORDERS.items():
                with self.subTest(sample=label, order=order_name):
                    compilations = [
                        compile_ast(copy.deepcopy(ast), order)[2]
                        for _ in range(3)
                    ]
                    self.assertEqual(compilations[0], compilations[1])
                    self.assertEqual(compilations[1], compilations[2])
                    # Exact bytes, not just Unicode string equality.
                    encoded = [text.encode("utf-8") for text in compilations]
                    self.assertEqual(encoded[0], encoded[1])
                    self.assertEqual(encoded[1], encoded[2])

    def test_same_order_runs_are_observably_deterministic(self):
        for label, builder, entry, arguments, _pin in _CASES:
            ast = builder()
            for order_name, order in LEGAL_ORDERS.items():
                _l, optimized, _t = compile_ast(ast, order)
                first = _interpret(optimized, entry, arguments)
                second = _interpret(
                    compile_ast(ast, order)[1], entry, arguments)
                self.assertEqual(
                    first, second,
                    msg=f"{label!r} / {order_name!r}: non-deterministic run",
                )


class IdempotenceTests(unittest.TestCase):
    def test_reapplying_same_sequence_is_a_noop(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            for order_name, order in IDEMPOTENT_ORDERS.items():
                with self.subTest(sample=label, order=order_name):
                    _l, once, text_once = compile_ast(ast, order)
                    self.assertTrue(once.ssa)

                    # Apply the very same sequence to the optimized result.
                    twice = apply_order(once, order)
                    text_twice = render_module(twice)
                    self.assertEqual(
                        text_once, text_twice,
                        msg=(f"{label!r} / {order_name!r}: canonical target "
                             "code changed on second application"),
                    )
                    self.assertEqual(
                        structural_signature(once),
                        structural_signature(twice),
                        msg=(f"{label!r} / {order_name!r}: canonical IR "
                             "structure changed on second application"),
                    )
                    self.assertIsNot(twice, once)

                    # A third application must be stable as well.
                    thrice = apply_order(twice, order)
                    self.assertEqual(
                        text_twice, render_module(thrice),
                        msg=f"{label!r} / {order_name!r}: not a fixed point",
                    )

    def test_individual_repeatable_passes_are_fixpoints(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            lowered = lower_module(copy.deepcopy(ast))
            ssa_once = to_ssa(lowered)
            ssa_twice = to_ssa(ssa_once)
            with self.subTest(sample=label, pass_="ssa"):
                self.assertEqual(
                    render_module(ssa_once), render_module(ssa_twice))
                self.assertEqual(
                    structural_signature(ssa_once),
                    structural_signature(ssa_twice),
                )

            dce_once = eliminate_dead_code(ssa_once)
            dce_twice = eliminate_dead_code(dce_once)
            with self.subTest(sample=label, pass_="dce"):
                self.assertEqual(
                    render_module(dce_once), render_module(dce_twice))
                self.assertEqual(
                    structural_signature(dce_once),
                    structural_signature(dce_twice),
                )

    def test_raw_core_reaches_canonical_fixed_point(self):
        # ssa,dce may leave numbering holes; the trailing canonicalizing ssa
        # pass is precisely what turns it into the default pipeline's stable
        # endpoint, and iterating ssa,dce from there changes nothing.
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            lowered, core, _text = compile_ast(ast, RAW_CORE_ORDER)
            # DCE alone is already a structural fixpoint even with holes.
            self.assertEqual(
                render_module(core),
                render_module(eliminate_dead_code(core)),
            )
            canonical = to_ssa(core)
            cycled = to_ssa(eliminate_dead_code(canonical))
            self.assertEqual(
                render_module(canonical), render_module(cycled),
                msg=(f"{label!r}: default-style endpoint is not the "
                     "canonical fixed point"),
            )

    def test_inputs_are_not_mutated_by_reapplication(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            lowered, once, text_before = compile_ast(ast, DEFAULT_ORDER)
            lowered_text_before = render_module(lowered)
            apply_order(once, DEFAULT_ORDER)
            self.assertEqual(render_module(once), text_before)
            self.assertEqual(render_module(lowered), lowered_text_before)


class PassOrderPreconditionTests(unittest.TestCase):
    def test_dce_before_ssa_is_rejected(self):
        lowered = lower_module(_clean_program())
        self.assertFalse(lowered.ssa)
        with self.assertRaises(ValueError):
            apply_order(lowered, ("dce",))
        with self.assertRaises(ValueError):
            apply_order(lowered, ("dce", "ssa"))
        # The pass itself enforces the same precondition directly.
        with self.assertRaises(ValueError):
            eliminate_dead_code(lowered)

    def test_instruction_selection_must_be_last(self):
        lowered = lower_module(_clean_program())
        with self.assertRaises(ValueError):
            apply_order(lowered, ("isel", "dce"))
        with self.assertRaises(ValueError):
            apply_order(lowered, ("ssa", "isel", "dce"))

    def test_unknown_pass_is_rejected(self):
        lowered = lower_module(_clean_program())
        with self.assertRaises(ValueError):
            apply_order(lowered, ("ssa", "fold"))

    def test_every_legal_order_keeps_ssa_before_dce_and_ir_before_isel(self):
        for name, order in LEGAL_ORDERS.items():
            seen_ssa = False
            for index, pass_name in enumerate(order):
                if pass_name == "ssa":
                    seen_ssa = True
                if pass_name == "dce":
                    self.assertTrue(
                        seen_ssa,
                        msg=f"legal order {name!r} runs dce before ssa",
                    )
            # Rendering (instruction selection) is always performed after.
            ast = _clean_program()
            _l, module, text = compile_ast(ast, order)
            self.assertTrue(text.endswith("\n"))
            self.assertIn("function clean", text)


class DceShapeOnFocusedFormsTests(unittest.TestCase):
    """Structural pins for the forms where passes genuinely interact."""

    def _caller(self, module: Module, name: str):
        return next(fn_ for fn_ in module.functions if fn_.name == name)

    def test_deletable_chains_removed_but_side_effecting_calls_kept(self):
        _l, optimized, _t = compile_ast(_clean_program(), DEFAULT_ORDER)
        caller = self._caller(optimized, "clean")
        calls = [ins.name for b in caller.blocks for ins in b.instructions
                 if isinstance(ins, Call)]
        self.assertEqual(calls, ["observe", "observe"])

        consts = {ins.value for b in caller.blocks
                  for ins in b.instructions if isinstance(ins, Const)}
        self.assertEqual(consts, {10, 99, 4})

        binops = [
            (ins.kind, ins.operator) for b in caller.blocks
            for ins in b.instructions if isinstance(ins, BinOp)
        ]
        self.assertEqual(binops, [("arith", "add"), ("arith", "add")])

    def test_inlineable_call_is_never_silently_removed(self):
        _l, optimized, _t = compile_ast(_around_program(), DEFAULT_ORDER)
        caller = self._caller(optimized, "around")
        calls = [ins.name for b in caller.blocks for ins in b.instructions
                 if isinstance(ins, Call)]
        self.assertEqual(calls, ["dbl", "after"])
        # Dead post-call arithmetic (v*9, +5) is gone; the live n+1 argument
        # add survives.
        binops = [
            ins.operator for b in caller.blocks
            for ins in b.instructions if isinstance(ins, BinOp)
        ]
        self.assertEqual(binops, ["add"])

    def test_join_phi_shapes_preserved_through_orders(self):
        # The trivial merge (a) prunes to no phi; the real merge (b) keeps a
        # phi in the merge block under every legal order.
        for order_name, order in LEGAL_ORDERS.items():
            _l, module, _t = compile_ast(_join_program(), order)
            caller = self._caller(module, "join")
            phi_blocks = {b.id: len(b.phis) for b in caller.blocks
                          if b.phis}
            # Exactly one block carries exactly one phi: b's merge.
            self.assertEqual(
                list(phi_blocks.values()), [1],
                msg=f"{order_name!r}: unexpected phi layout {phi_blocks}",
            )
            [merge_block] = [b for b in caller.blocks if b.phis]
            [phi] = merge_block.phis
            self.assertEqual(len(phi.entries), 2)


class ExistingEntryContractTests(unittest.TestCase):
    """The ordering harness enters through the same public compiler API and
    inherits its existing diagnostics and default flow."""

    def test_type_error_still_rejected_before_any_order(self):
        bad = program(func(
            "f", [], "int", [ret(arith("add", int_(1), bool_(False)))]))
        with self.assertRaises(TypeCheckError):
            lower_module(bad)

    def test_undefined_symbol_still_rejected(self):
        bad = program(func("f", [], "int", [ret(var("ghost"))]))
        with self.assertRaises(UndefinedSymbolError):
            lower_module(bad)

    def test_default_flow_text_remains_deterministic_and_renderable(self):
        _l, optimized, text = compile_ast(_loop_program(), DEFAULT_ORDER)
        self.assertEqual(
            text,
            compile_ast(_loop_program(), DEFAULT_ORDER)[2],
        )
        # SSA target format markers are unchanged.
        self.assertIn("function loopsum(", text)
        self.assertNotIn("%t", text)
        self.assertNotIn("locals:", text)


if __name__ == "__main__":
    unittest.main()
