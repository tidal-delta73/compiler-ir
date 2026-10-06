"""Pass-ordering, fixpoint and stage-contract regression tests.

With all five public pieces in place -- ``lower_module``, ``to_ssa``,
``fold_constants``, ``eliminate_dead_code`` and ``render_module``, each
already pinned by its own suite -- this module treats the compiler as a
*pipeline* made of those existing public entry points and checks that:

1. One legal source program, entered through the existing public AST entry
   (:func:`lower_module`, which still runs AST validation, name resolution
   and type checking), produces the same observable behavior under the
   default optimization pipeline and every alternative *legal* pass order
   -- fold/DCE order, repetitions and interleavings -- using the
   unoptimized (non-SSA) module's execution as the semantic baseline.
2. For fixed AST, arguments and order, repeated compilations are
   byte-identical (determinism); different orders need only agree on
   observable behavior -- never on definition count, SSA numbering or
   emitted text.
3. The individually repeated passes (``to_ssa``, ``fold_constants``,
   ``eliminate_dead_code``) are each a structural and textual fixpoint;
   for complete schedules that include folding and elimination, a second
   application is asserted idempotent only once it has actually reached
   the normalized endpoint, and every call returns an independent object
   without mutating its input.
4. Orderings may only permute passes inside the existing stage contracts:
   SSA construction precedes every SSA-dependent optimization
   (``fold_const``/``dce``), and instruction selection (the deterministic
   :func:`render_module` emission) runs after IR optimization.

The samples collectively cover cross-basic-block propagation; same- and
different-constant phis at a branch convergence; a loop header phi;
foldable arithmetic and comparisons; definitions that become dead only
after folding; and paths whose result is unused but whose calls remain
observable side effects.

Passes and preconditions (nothing new is added to the compiler package)
-----------------------------------------------------------------------

* ``ssa``   -- :func:`to_ssa`.  Legal on a lowered module and, as an
  idempotent canonicalization (clone + deterministic renumbering), on an
  already-SSA module.
* ``fold``  -- :func:`fold_constants`.  Its documented precondition is an
  SSA module: scheduling it before ``ssa`` raises ``ValueError``.
* ``dce``   -- :func:`eliminate_dead_code`.  Its documented precondition is
  an SSA module: scheduling it before ``ssa`` raises ``ValueError``.
* instruction selection is the final :func:`render_module` and is fixed
  last.

Observables
-----------

The language has no I/O statement, so the ordered trace of executed calls
``(callee, positional args)`` is the standard-output surrogate -- the only
side effect the language can have (it is also the observable used by
``test_semantic_equivalence.py``).  An outcome is one of:

* normal termination: return value (``None`` for void) + call trace;
* a runtime fault: error category, trigger site, the operands at the
  fault, and the call trace produced *before* the fault.

The language's grammar already includes ``div``/``mod``; the single
runtime error the language defines is a zero divisor.  Division and
modulo follow truncation toward zero (the remainder satisfies
``a == (a/b)*b + a%b``).  There is no array or indexing construct, so no
bounds error exists in this language subset, and no undefined-behavior
sample is used: every faulting instruction's result is consumed (returned,
branched on or accumulated), so no legal order can legally delete it.

A fault site is identified independently of SSA numbering by
``(function name, block label, operator, ordinal of the BinOp among the
BinOps of its block)``: block labels are assigned by the lowering DFS and
preserved by ``to_ssa``, folding and DCE, and the relative order of
surviving BinOps is preserved as well.  Every faulting block in this suite
contains exactly one BinOp -- constant folding turns the *other*
arithmetic into Consts, which never occurs inside the faulting block --
so the site's ordinal is 0 and is unambiguous under every order.

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
    fold_constants,
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
            # BinOps (fold/DCE preserve surviving relative order; folded
            # BinOps become Consts and therefore leave this enumeration).
            # BinOp nodes are mutable dataclasses and therefore unhashable,
            # so key by id.
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

    _assert(False, f"unknown BinOp kind {instruction.kind!r}")


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


# Full schedule: SSA construction, constant folding, dead-code elimination,
# then a final SSA canonicalization (clone + deterministic renumbering),
# which makes the emitted IR the fixed point of the whole sequence: fold/DCE
# may leave numbering holes, and the trailing ssa pass compacts them without
# changing structure or semantics.  fold precedes dce on purpose: folding
# exposes dead definitions that DCE then reclaims.
DEFAULT_ORDER = ("ssa", "fold", "dce", "ssa")

# Full legal alternatives: a repeated fold, DCE before the fold/fold->DCE
# cleanup pair, a repeated DCE, and a fold/dce interleaving.  Every one ends
# in the canonicalizing ssa pass, so each is a fixed point of its own
# sequence and all five reach byte-identical canonical IR.
ALT_FOLD_REPEAT = ("ssa", "fold", "fold", "dce", "ssa")
ALT_DCE_BEFORE_FOLD = ("ssa", "dce", "fold", "dce", "ssa")
ALT_DCE_REPEAT = ("ssa", "fold", "dce", "dce", "ssa")
ALT_INTERLEAVE = ("ssa", "fold", "dce", "fold", "dce", "ssa")

# Legal but deliberately less cleaning: no elimination at all (fold only,
# fold + canonicalization), no fold (construction only), and the raw cores
# without the trailing canonicalization.  These keep different numbers of
# definitions and different SSA numbers, and are compared on observables
# rather than text.
NO_DCE_FOLD_ONLY = ("ssa", "fold")
NO_DCE_FOLD_CANON = ("ssa", "fold", "ssa")
NO_OPT_ORDER = ("ssa",)
RAW_FOLD_DCE = ("ssa", "fold", "dce")
RAW_DCE_FOLD = ("ssa", "dce", "fold")

LEGAL_ORDERS = {
    "default ssa,fold,dce,ssa": DEFAULT_ORDER,
    "alt repeated fold ssa,fold,fold,dce,ssa": ALT_FOLD_REPEAT,
    "alt dce-before-fold ssa,dce,fold,dce,ssa": ALT_DCE_BEFORE_FOLD,
    "alt repeated dce ssa,fold,dce,dce,ssa": ALT_DCE_REPEAT,
    "alt interleave ssa,fold,dce,fold,dce,ssa": ALT_INTERLEAVE,
    "no-elimination fold-only ssa,fold": NO_DCE_FOLD_ONLY,
    "no-elimination fold+canon ssa,fold,ssa": NO_DCE_FOLD_CANON,
    "no-optimization ssa": NO_OPT_ORDER,
    "raw core ssa,fold,dce": RAW_FOLD_DCE,
    "raw core ssa,dce,fold": RAW_DCE_FOLD,
}

# The five complete schedules: each contains a fold and a dce, ends with
# ssa, and reaches the same normalized endpoint -- strict idempotence and
# canonical-text agreement are asserted only for these.
FULL_ORDERS = {
    name: order for name, order in LEGAL_ORDERS.items()
    if name.startswith(("default", "alt"))
}

# Partial schedules whose own sequence, re-applied to its endpoint, needs a
# second round to settle (their endpoints carry fold-made-dead defs or
# numbering holes).  They are idempotent only *after* converging.
NON_FIXPOINT_PARTIALS = {
    name: order for name, order in LEGAL_ORDERS.items()
    if name.startswith("raw core")
}

# Ordering harness pass names -> existing public entry points.
_PASSES = {
    "ssa": to_ssa,
    "fold": fold_constants,
    "dce": eliminate_dead_code,
}


def apply_order(module: Module, order) -> Module:
    """Apply ``order`` (a tuple of pass names) to ``module``.

    :raises ValueError: if a pass name is unknown, if instruction selection
        appears anywhere but the end, or if ``fold``/``dce`` is scheduled
        before the first ``ssa`` (both require an SSA module).
    """
    ssa_seen = bool(getattr(module, "ssa", False))
    for name in order:
        if name == "ssa":
            module = to_ssa(module)
            ssa_seen = True
        elif name in ("fold", "dce"):
            if not ssa_seen:
                raise ValueError(
                    "pass order violates precondition: "
                    f"{name!r} requires an SSA module, but no 'ssa' pass "
                    "precedes it"
                )
            module = _PASSES[name](module)
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


def _definition_count(module: Module, function_name: str) -> int:
    fn_ = next(f for f in module.functions if f.name == function_name)
    return (
        len(fn_.params)
        + sum(len(b.phis) for b in fn_.blocks)
        + sum(len(b.instructions) for b in fn_.blocks)
    )


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
# their constants) -- in the canonical schedules constant folding first
# turns most of the chain into Consts and DCE then reclaims them -- but
# BOTH calls remain roots even though `ignored`'s result is never read:
# deleting either would drop observable output.
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
# Folding only collapses compile-time-constant subexpressions; the header
# phi whose back edge disagrees with its entry edge is left unfolded.
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
# under DCE (folded into Consts first in the fold-containing schedules),
# while the later side-effecting call `after` is preserved.
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


# F) Definitions exposed as dead only after constant folding.
#
# `p`/`q`/`s` are a fully constant chain (20 + 22 = 42) whose result feeds
# a live subtraction s-n: without folding the chain is needed, so DCE
# *before* folding cannot remove its inner arithmetic; after folding the
# chain is Consts and the same schedule's cleanup DCE reclaims every
# intermediate.  `a` arrives at the merge as the same folded literal 7 on
# both arms (3+4 and 10-3), so its merge phi folds to a Const and is then
# dead; the call consuming `a` stays an observable root regardless.
def _fold_exposed_program():
    return program(
        func("exposed", [param("c", "bool"), param("n", "int")], "int", [
            let("p", "int", int_(20)),
            let("q", "int", int_(22)),
            let("s", "int", arith("add", var("p"), var("q"))),
            let("a", "int", int_(0)),
            if_(var("c"),
                [assign("a", arith("add", int_(3), int_(4)))],
                [assign("a", arith("sub", int_(10), int_(3)))]),
            let("e", "int", call("emit", [var("a")])),
            ret(arith("add", arith("sub", var("s"), var("n")), var("e"))),
        ]),
        _emit_function(),
    )


# G) Same-constant and different-constant phis at one branch convergence.
#
# `same` is 2+3 and 8-3 on its two edges: both fold to the literal 5, so
# the merge phi folds to Const 5 and disappears.  `diff` is genuinely 11
# vs 12: its phi survives every order and feeds the observable call.  The
# returned value combines both, so the folded value is itself used.
def _phi_merge_program():
    return program(
        func("phimerge", [param("c", "bool")], "int", [
            let("same", "int", int_(0)),
            if_(var("c"),
                [assign("same", arith("add", int_(2), int_(3)))],
                [assign("same", arith("sub", int_(8), int_(3)))]),
            let("diff", "int", int_(0)),
            if_(var("c"),
                [assign("diff", int_(11))],
                [assign("diff", int_(12))]),
            let("r", "int", call("emit", [var("diff")])),
            ret(arith("add", var("same"), var("r"))),
        ]),
        _emit_function(),
    )


# H) A loop header phi with a constant entry edge and a folded step.
#
# i starts at 0 and advances by (1+5) = 6 per round.  The step arithmetic
# folds, but the header phi looks constant only on the entry edge: the
# lattice fixpoint must leave it (and the i+6 add) unfolded once the back
# edge disagrees.  No calls occur, so this pins loop-carried values.
def _loop_step_program():
    return program(
        func("loopstep", [param("n", "int")], "int", [
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")),
                   [assign("i", arith(
                       "add", var("i"), arith("add", int_(1), int_(5))))]),
            ret(var("i")),
        ]),
    )


# I) A foldable comparison steering a branch; both arms' calls stay roots.
#
# 6 == 6 folds to the literal true, but the pass must not rewrite the
# constant Branch into a Jump or delete either block: interpreting the
# module still takes the true arm.  Both calls are roots in every order
# (results unused), pinning "unused result but observable side effect".
def _constant_flag_program():
    return program(
        func("constflag", [], "void", [
            let("flag", "bool", compare("eq", int_(6), int_(6))),
            if_(var("flag"),
                [let("r1", "int", call("emit", [int_(1)]))],
                [let("r2", "int", call("emit", [int_(2)]))]),
        ]),
        _emit_function(),
    )


# J) A zero divisor whose value is discovered ONLY by folding.
#
# Both arms compute (3*2) - 6 = 0 through folded arithmetic and a
# same-literal merge phi.  Without folding the divisor SSA value is not
# statically known; folding still must NOT fold the faulting div/mod away
# (its result is returned, and the trap has to fire at runtime after the
# preceding emit).  The faulting BinOp is the sole BinOp in the merge
# block b3, so its numbering-independent ordinal stays 0 even though the
# arm arithmetic folds to Consts.
def _folded_zero_program(operator, function_name):
    return program(
        func(function_name, [param("c", "bool")], "int", [
            let("six", "int", arith("mul", int_(3), int_(2))),
            let("z", "int", int_(1)),
            if_(var("c"),
                [assign("z", arith("sub", var("six"), int_(6)))],
                [assign("z", arith("sub", var("six"), int_(6)))]),
            let("e", "int", call("emit", [int_(7)])),
            let("q", "int", arith(operator, int_(100), var("z"))),
            ret(arith("add", var("q"), var("e"))),
        ]),
        _emit_function(),
    )


# (label, AST builder, entry, argument tuple, pinned expected Outcome) -- the
# pin anchors ground truth independently of the non-SSA baseline cross-check.
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
            # then the division (ordinal 1) that actually faults.  The
            # divisor sub depends on the parameter i and never folds.
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
    (
        "dead definitions exposed by folding",
        _fold_exposed_program, "exposed", (True, 5),
        Outcome.normal(44, [("emit", (7,))]),
    ),
    (
        "dead definitions exposed by folding (other edge, zero offset)",
        _fold_exposed_program, "exposed", (False, 0),
        Outcome.normal(49, [("emit", (7,))]),
    ),
    (
        "same- and different-constant merge phis",
        _phi_merge_program, "phimerge", (True,),
        Outcome.normal(16, [("emit", (11,))]),
    ),
    (
        "same- and different-constant merge phis (other edge)",
        _phi_merge_program, "phimerge", (False,),
        Outcome.normal(17, [("emit", (12,))]),
    ),
    (
        "loop header phi with folded step",
        _loop_step_program, "loopstep", (4,),
        Outcome.normal(6, []),
    ),
    (
        "loop header phi with folded step (boundary)",
        _loop_step_program, "loopstep", (6,),
        Outcome.normal(6, []),
    ),
    (
        "loop header phi with folded step (zero trips)",
        _loop_step_program, "loopstep", (0,),
        Outcome.normal(0, []),
    ),
    (
        "foldable comparison steering a branch",
        _constant_flag_program, "constflag", (),
        Outcome.normal(None, [("emit", (1,))]),
    ),
    (
        "folded same-constant phi reveals division by zero",
        lambda: _folded_zero_program("div", "foldzero"),
        "foldzero", (True,),
        Outcome(
            "fault",
            output=[("emit", (7,))],
            category=ZERO_DIVISOR,
            site=("foldzero", "b3", "div", 0),
            operands=(100, 0),
        ),
    ),
    (
        "folded same-constant phi reveals modulo by zero",
        lambda: _folded_zero_program("mod", "foldzero"),
        "foldzero", (False,),
        Outcome(
            "fault",
            output=[("emit", (7,))],
            category=ZERO_DIVISOR,
            site=("foldzero", "b3", "mod", 0),
            operands=(100, 0),
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

    def test_different_orders_keep_different_text_but_same_semantics(self):
        # Semantic equivalence must never be established from identical
        # text: the fold-exposed sample keeps far more definitions without
        # elimination than under the canonical schedule (its constant
        # chains are not yet reclaimed), so its text differs while its
        # return value and call order agree for every argument set.
        ast = _fold_exposed_program()
        _l, canonical, canon_text = compile_ast(ast, DEFAULT_ORDER)
        _l2, folded_only, fold_text = compile_ast(ast, NO_DCE_FOLD_ONLY)
        _l3, plain_ssa, ssa_text = compile_ast(ast, NO_OPT_ORDER)

        self.assertNotEqual(ssa_text, fold_text)
        self.assertNotEqual(fold_text, canon_text)
        self.assertGreater(
            _definition_count(folded_only, "exposed"),
            _definition_count(canonical, "exposed"),
        )
        # Folding rewrites slots in place (BinOp -> Const, folded phi -> a
        # leading Const) without deleting definitions, so the counts can be
        # equal even though the text differs; order never adds definitions.
        self.assertGreaterEqual(
            _definition_count(plain_ssa, "exposed"),
            _definition_count(folded_only, "exposed"),
        )

        for arguments in ((True, 5), (False, 0), (True, 49)):
            expected = _interpret(
                lower_module(copy.deepcopy(ast)), "exposed", arguments)
            for module in (plain_ssa, folded_only, canonical):
                self.assertEqual(
                    _interpret(module, "exposed", arguments), expected,
                    msg="text layouts differ by design; outcomes must not",
                )

    def test_fault_category_site_prefix_and_operands_agree_across_orders(self):
        fault_cases = [case for case in _CASES if case[4].kind == "fault"]
        self.assertTrue(fault_cases, "suite must include runtime-fault cases")
        for label, builder, entry, arguments, pinned in fault_cases:
            ast = builder()
            baseline = _interpret(
                lower_module(copy.deepcopy(ast)), entry, arguments)
            self.assertEqual(baseline.kind, "fault")
            self.assertEqual(baseline.category, pinned.category)
            self.assertEqual(baseline.site, pinned.site)
            self.assertEqual(baseline.operands, pinned.operands)
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

    def test_folding_does_not_swallow_or_advance_the_zero_trap(self):
        # The folded-zero samples: the divisor is provably zero only after
        # folding across the same-literal phi, yet every order keeps exactly
        # one div/mod BinOp at b3 and the emit precedes the fault.  A raw
        # fold (no DCE) must already keep the fault -- folding alone never
        # deletes the trap.
        for operator, arguments in (("div", (True,)), ("mod", (False,))):
            ast = _folded_zero_program(operator, "foldzero")
            for order_name in ("no-optimization ssa",
                               "no-elimination fold-only ssa,fold",
                               "raw core ssa,fold,dce",
                               "default ssa,fold,dce,ssa"):
                with self.subTest(operator=operator, order=order_name):
                    _l, module, _t = compile_ast(
                        ast, LEGAL_ORDERS[order_name])
                    merge = next(
                        b for b in module.functions[0].blocks
                        if b.label == "b3")
                    traps = [ins for ins in merge.instructions
                             if isinstance(ins, BinOp)
                             and ins.operator == operator]
                    self.assertEqual(len(traps), 1)
                    outcome = _interpret(module, "foldzero", arguments)
                    self.assertEqual(outcome.kind, "fault")
                    self.assertEqual(outcome.category, ZERO_DIVISOR)
                    self.assertEqual(outcome.site,
                                     ("foldzero", "b3", operator, 0))
                    self.assertEqual(outcome.operands, (100, 0))
                    self.assertEqual(outcome.output, [("emit", (7,))])


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


class FixpointAndIdempotenceTests(unittest.TestCase):
    def test_full_schedules_are_idempotent_and_share_canonical_text(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            canonical_text = None
            for order_name, order in FULL_ORDERS.items():
                with self.subTest(sample=label, order=order_name):
                    _l, once, text_once = compile_ast(ast, order)
                    self.assertTrue(once.ssa)

                    # Re-applying the very same (already complete) sequence
                    # to the optimized result changes neither structure nor
                    # target text, and returns a fresh object.
                    twice = apply_order(once, order)
                    self.assertIsNot(twice, once)
                    self.assertEqual(render_module(twice), text_once)
                    self.assertEqual(
                        structural_signature(twice),
                        structural_signature(once),
                    )
                    # A third application must be stable as well.
                    thrice = apply_order(twice, order)
                    self.assertEqual(
                        render_module(thrice), text_once,
                        msg=f"{label!r} / {order_name!r}: not a fixed point",
                    )

                    # All complete schedules normalize to the same bytes.
                    if canonical_text is None:
                        canonical_text = text_once
                    else:
                        self.assertEqual(
                            text_once, canonical_text,
                            msg=(f"{label!r} / {order_name!r}: complete "
                                 "schedules disagree on canonical text"),
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
                self.assertIsNot(ssa_twice, ssa_once)

            fold_once = fold_constants(ssa_once)
            fold_twice = fold_constants(fold_once)
            with self.subTest(sample=label, pass_="fold"):
                self.assertEqual(
                    render_module(fold_once), render_module(fold_twice))
                self.assertEqual(
                    structural_signature(fold_once),
                    structural_signature(fold_twice),
                )
                self.assertIsNot(fold_twice, fold_once)

            dce_once = eliminate_dead_code(ssa_once)
            dce_twice = eliminate_dead_code(dce_once)
            with self.subTest(sample=label, pass_="dce"):
                self.assertEqual(
                    render_module(dce_once), render_module(dce_twice))
                self.assertEqual(
                    structural_signature(dce_once),
                    structural_signature(dce_twice),
                )
                self.assertIsNot(dce_twice, dce_once)

    def test_raw_cores_become_idempotent_only_after_reaching_endpoint(self):
        # ssa,fold,dce and ssa,dce,fold lack the trailing canonicalization
        # and (for dce,fold) the post-fold cleanup, so re-applying the same
        # sequence may first have to settle them -- for dce,fold, folding
        # exposes dead constants that the next round's DCE reclaims before
        # the numbering stabilizes.  Idempotence is therefore asserted only
        # once iteration has actually converged, never from the first repeat.
        convergence_rounds = {"raw core ssa,fold,dce": set(),
                              "raw core ssa,dce,fold": set()}
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            for order_name, order in NON_FIXPOINT_PARTIALS.items():
                with self.subTest(sample=label, order=order_name):
                    _l, endpoint, _text = compile_ast(ast, order)

                    # Drive the sequence to its fixed point, recording how
                    # many re-applications that took (must terminate quickly
                    # -- every round is subtractive or a renumbering).
                    current = endpoint
                    current_text = render_module(current)
                    rounds = 0
                    for rounds in range(1, 6):
                        nxt = apply_order(current, order)
                        nxt_text = render_module(nxt)
                        if nxt_text == current_text:
                            settled, settled_text = nxt, nxt_text
                            break
                        current, current_text = nxt, nxt_text
                    else:  # pragma: no cover - defensive
                        self.fail("raw core did not converge within 5 rounds")
                    convergence_rounds[order_name].add(rounds)

                    # Now that the normalized endpoint is reached, further
                    # applications are a structural/textual no-op and return
                    # fresh SSA objects.
                    again = apply_order(settled, order)
                    self.assertTrue(again.ssa)
                    self.assertIsNot(again, settled)
                    self.assertEqual(render_module(again), settled_text)
                    self.assertEqual(
                        structural_signature(again),
                        structural_signature(settled),
                    )

                    # Completing with the standard tail (dce,ssa) reaches
                    # the default pipeline's canonical text.
                    completed = to_ssa(eliminate_dead_code(settled))
                    _l2, canonical, _t2 = compile_ast(ast, DEFAULT_ORDER)
                    self.assertEqual(
                        render_module(completed), render_module(canonical))

        # The dce-first raw core genuinely needed a non-trivial resettle for
        # at least one fold-exposing sample; that is the interaction this
        # guard exists to pin.
        self.assertIn(2, convergence_rounds["raw core ssa,dce,fold"])

    def test_every_call_returns_independent_object_and_keeps_input(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            lowered, once, text_before = compile_ast(ast, DEFAULT_ORDER)
            lowered_text_before = render_module(lowered)
            # The re-application returns a distinct object...
            self.assertIsNot(apply_order(once, DEFAULT_ORDER), once)
            # ...and neither endpoint nor the original lowered module moved.
            self.assertEqual(render_module(once), text_before)
            self.assertEqual(render_module(lowered), lowered_text_before)

            # Individual pass independence/non-mutation on representative
            # inputs (fold + dce on the raw SSA module).
            ssa = to_ssa(lower_module(copy.deepcopy(ast)))
            ssa_text = render_module(ssa)
            folded = fold_constants(ssa)
            self.assertIsNot(folded, ssa)
            self.assertEqual(render_module(ssa), ssa_text)
            cleaned = eliminate_dead_code(ssa)
            self.assertIsNot(cleaned, ssa)
            self.assertEqual(render_module(ssa), ssa_text)

    def test_no_pass_ever_marks_or_mutates_into_non_ssa(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            for order_name, order in LEGAL_ORDERS.items():
                _l, module, _t = compile_ast(ast, order)
                with self.subTest(sample=label, order=order_name):
                    self.assertTrue(module.ssa)
                    self.assertTrue(all(f.ssa for f in module.functions))


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

    def test_fold_before_ssa_is_rejected(self):
        lowered = lower_module(_clean_program())
        self.assertFalse(lowered.ssa)
        with self.assertRaises(ValueError):
            apply_order(lowered, ("fold",))
        with self.assertRaises(ValueError):
            apply_order(lowered, ("fold", "ssa"))
        with self.assertRaises(ValueError):
            apply_order(lowered, ("dce", "fold", "ssa"))
        # The pass itself enforces the same precondition directly.
        with self.assertRaises(ValueError):
            fold_constants(lowered)

    def test_instruction_selection_must_be_last(self):
        lowered = lower_module(_clean_program())
        with self.assertRaises(ValueError):
            apply_order(lowered, ("isel", "fold"))
        with self.assertRaises(ValueError):
            apply_order(lowered, ("ssa", "isel", "fold"))
        with self.assertRaises(ValueError):
            apply_order(lowered, ("ssa", "fold", "isel", "dce"))

    def test_unknown_pass_is_rejected(self):
        lowered = lower_module(_clean_program())
        for bad in ("unknown", "inline", "simplify", "FOLD"):
            with self.assertRaises(ValueError):
                apply_order(lowered, ("ssa", bad))

    def test_every_legal_order_keeps_ssa_first_and_render_last(self):
        for name, order in LEGAL_ORDERS.items():
            seen_ssa = False
            for pass_name in order:
                if pass_name == "ssa":
                    seen_ssa = True
                if pass_name in ("fold", "dce"):
                    self.assertTrue(
                        seen_ssa,
                        msg=f"legal order {name!r} runs {pass_name} "
                            "before ssa",
                    )
            # Rendering (instruction selection) is always performed after.
            ast = _clean_program()
            _l, module, text = compile_ast(ast, order)
            self.assertTrue(text.endswith("\n"))
            self.assertIn("function clean", text)


class FoldDceInteractionShapeTests(unittest.TestCase):
    """Structural pins for the forms where folding and DCE interact."""

    @staticmethod
    def _caller(module: Module, name: str):
        return next(fn_ for fn_ in module.functions if fn_.name == name)

    def test_fold_exposed_dead_chain_is_reclaimed_only_with_dce(self):
        # Without elimination the folded constants remain; the canonical
        # schedule reclaims the 20/22 chain and the folded phi, leaving just
        # the live constant 42 (s), the merge constant 7 (a), the call and
        # the two live arithmetic ops (s-n and +e).
        _l, folded_only, _t = compile_ast(
            _fold_exposed_program(), NO_DCE_FOLD_ONLY)
        caller = self._caller(folded_only, "exposed")
        self.assertIn(
            20, {i.value for b in caller.blocks for i in b.instructions
                 if isinstance(i, Const)})

        _l2, canonical, _t2 = compile_ast(
            _fold_exposed_program(), DEFAULT_ORDER)
        caller = self._caller(canonical, "exposed")
        consts = sorted(
            i.value for b in caller.blocks
            for i in b.instructions if isinstance(i, Const))
        self.assertEqual(consts, [7, 42])
        self.assertEqual(
            [(i.kind, i.operator) for b in caller.blocks for i in b.instructions
             if isinstance(i, BinOp)],
            [("arith", "sub"), ("arith", "add")],
        )
        # The same-literal merge phi folded away in every fold-containing
        # order; no phi survives canonicalization.
        self.assertFalse(
            [phi for b in caller.blocks for phi in b.phis])
        # Side-effecting call retained exactly once.
        self.assertEqual(
            [i.name for b in caller.blocks for i in b.instructions
             if isinstance(i, Call)],
            ["emit"],
        )

    def test_same_constant_phi_folds_different_constant_phi_survives(self):
        _l, module, _t = compile_ast(_phi_merge_program(), DEFAULT_ORDER)
        caller = self._caller(module, "phimerge")
        phi_blocks = {b.id: [phi for phi in b.phis]
                      for b in caller.blocks if b.phis}
        # Exactly one block carries exactly one phi: the 11/12 merge.
        self.assertEqual(len(phi_blocks), 1)
        [phis] = phi_blocks.values()
        self.assertEqual(len(phis), 1)
        self.assertEqual(len(phis[0].entries), 2)
        # The same-literal merge folded to Const 5 and is itself used by the
        # live return arithmetic.
        self.assertIn(
            5, {i.value for b in caller.blocks for i in b.instructions
                 if isinstance(i, Const)})
        self.assertEqual(
            [i.name for b in caller.blocks for i in b.instructions
             if isinstance(i, Call)],
            ["emit"],
        )

    def test_loop_header_phi_survives_while_step_folds(self):
        _l, module, _t = compile_ast(_loop_step_program(), DEFAULT_ORDER)
        caller = self._caller(module, "loopstep")
        # The header (b1) keeps exactly one phi with two incoming edges.
        header = next(b for b in caller.blocks if b.label == "b1")
        self.assertEqual(len(header.phis), 1)
        self.assertEqual(len(header.phis[0].entries), 2)
        # The body keeps the i+6 add but the 1+5 step folded to Const 6,
        # and the header comparison stays a comparison.
        body = next(b for b in caller.blocks if b.label == "b2")
        self.assertTrue(any(
            isinstance(i, Const) and i.value == 6 for i in body.instructions))
        self.assertEqual(
            [i.operator for i in body.instructions
             if isinstance(i, BinOp)],
            ["add"],
        )
        self.assertIsInstance(
            header.terminator, Branch)

    def test_folded_comparison_branches_but_is_not_jump_rewritten(self):
        _l, module, _t = compile_ast(
            _constant_flag_program(), DEFAULT_ORDER)
        caller = self._caller(module, "constflag")
        # No block was deleted: condition, true, false and the reachable
        # post-if merge all remain, and the condition terminator is still a
        # Branch on a folded bool Const.
        self.assertEqual(len(caller.blocks), 4)
        self.assertIsInstance(caller.entry.terminator, Branch)
        cond = caller.entry.terminator.cond
        owner = next(i for b in caller.blocks for i in b.instructions
                     if i.dest is cond)
        self.assertIsInstance(owner, Const)
        self.assertEqual(owner.value, True)
        # Both arm calls are present as roots despite unused results; at
        # runtime only the true arm's call fires.
        self.assertEqual(
            [i.name for b in caller.blocks for i in b.instructions
             if isinstance(i, Call)],
            ["emit", "emit"],
        )
        outcome = _interpret(module, "constflag", ())
        self.assertEqual(outcome, Outcome.normal(None, [("emit", (1,))]))

    def test_deletable_chains_removed_but_side_effecting_calls_kept(self):
        _l, optimized, _t = compile_ast(_clean_program(), DEFAULT_ORDER)
        caller = self._caller(optimized, "clean")
        calls = [ins.name for b in caller.blocks for ins in b.instructions
                 if isinstance(ins, Call)]
        self.assertEqual(calls, ["observe", "observe"])

        consts = {ins.value for b in caller.blocks
                  for ins in b.instructions if isinstance(ins, Const)}
        # The fold-first canonical schedule propagates the constants, but
        # only the call argument 99 and the live +4 (and n+10's side) are
        # roots; the pure (1+2)*3 chain is gone.
        self.assertNotIn(1, consts)
        self.assertNotIn(2, consts)
        self.assertNotIn(3, consts)
        self.assertIn(99, consts)
        self.assertIn(4, consts)

    def test_join_phi_shapes_preserved_through_orders(self):
        # The trivial merge (a) prunes to no phi; the real merge (b) keeps a
        # phi in the merge block under every legal order.
        for order_name, order in LEGAL_ORDERS.items():
            _l, module, _t = compile_ast(_join_program(), order)
            caller = self._caller(module, "join")
            phi_blocks = {b.id: len(b.phis) for b in caller.blocks
                          if b.phis}
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
