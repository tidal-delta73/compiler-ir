"""Pass-ordering and idempotency regression tests for the optimizer.

Nothing here adds a language feature or touches the public entry points,
the default pipeline, the diagnostics or the emitted text format; this
module only *observes* the existing pipeline from more angles.

The pipeline under test maps the spec vocabulary onto this repository as
follows:

* front end (type checking + AST -> IR lowering): :func:`lower_module`;
* SSA construction: :func:`to_ssa`;
* SSA optimization: :func:`eliminate_dead_code` (the only IR-level
  optimization pass; constant folding/propagation do not exist here, so
  the repeatability checks target the passes that do);
* final emission ("target code"): :func:`render_module`, the fixed last
  stage of every ordering -- no variant moves a pass after it.

Legal orderings
---------------
A pass ordering is legal when it honours the existing stage constraints:
SSA construction precedes the SSA-dependent DCE pass, and rendering stays
last.  Within those constraints this suite exercises the default order
``lower -> ssa -> dce -> render`` plus the legal variants ``ssa``,
``ssa+dce+dce``, ``ssa+ssa+dce`` and ``ssa+dce+ssa+dce`` (both ``to_ssa``
and ``eliminate_dead_code`` accept their own output again).

What is compared
----------------
* Same source + same order, compiled repeatedly (in-process, across
  processes and across hash seeds): the emitted text must be
  byte-identical.
* Across *different* legal orders only observable semantics must agree --
  the return value, the ordered call trace (this language has no stdout;
  the call trace is its observable output channel) and, for the defined
  runtime error of the language (division/modulus by zero), the error
  category and the trap location ``(function, block)``.  Textual or
  structural equality across orders is never used as a correctness
  criterion, and one test pins that the orders genuinely differ in text.
* Idempotency: applying ``to_ssa`` / ``eliminate_dead_code`` to their own
  result changes nothing, a second DCE performs no new structural
  removals, and the default sequence is a fixed point on its own
  renormalized result (DCE keeps surviving SSA numbers, so one
  re-normalizing ``to_ssa`` compacts the holes exactly once).
* The unoptimized (lower-only) module is the semantic baseline; every
  expected value is also pinned literally so the baseline itself is
  anchored.  All programs are fully defined -- no undefined-behaviour
  sample takes part in the semantic comparison.

Every failure message names the source sample, the pass order and the
first divergent observable result.  All inputs are fixed; nothing uses
randomness, wall-clock time or hash-order-dependent iteration.
"""
import json
import os
import subprocess
import sys
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


# --------------------------------------------------------------------------
# IR interpreter with defined runtime errors
# --------------------------------------------------------------------------
#
# Mirrors the interpreter in test_semantic_equivalence.py but covers the
# full arithmetic operator set of the language (add/sub/mul/div/mod) and
# reports the language's defined runtime error -- division or modulus by
# zero -- as a _Trap carrying its category and source location.


_STEP_LIMIT = 100_000

_ARITH = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}

_COMPARE = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def _trunc_div(a, b):
    """Truncating (C-like) integer division; deterministic for all signs."""
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def _trunc_mod(a, b):
    return a - _trunc_div(a, b) * b


class _Trap(Exception):
    """A defined runtime error with its category and source location."""

    def __init__(self, category, function, block):
        super().__init__(f"{category} in {function} at {block}")
        self.category = category
        self.function = function
        self.block = block


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


class _Interpreter:
    """Execute a lowered :class:`Module` of either SSA flavor.

    ``run`` returns the entry function's return value and appends one
    ``(callee, arguments)`` tuple per executed call to ``trace``.  A
    division or modulus by zero raises :class:`_Trap` with the category
    and the ``(function, block)`` location of the faulting instruction.
    """

    def __init__(self, module):
        _assert(isinstance(module, Module),
                f"interpreter expects a Module, got {type(module).__name__}")
        self.ssa = bool(module.ssa)
        self._functions = {fn.name: fn for fn in module.functions}

    def run(self, entry, arguments, trace):
        fn = self._functions.get(entry)
        _assert(fn is not None, f"unknown entry function {entry!r}")
        arguments = tuple(arguments)
        _assert(
            len(arguments) == len(fn.params),
            f"entry {entry!r} expects {len(fn.params)} argument(s), "
            f"got {len(arguments)}",
        )
        return self._invoke(fn, arguments, trace)

    def _invoke(self, fn, arguments, trace):
        definitions = {}
        slots = {}
        temps = {}

        for parameter, argument in zip(fn.params, arguments):
            if self.ssa:
                _assert(parameter.temp is not None,
                        f"SSA parameter {parameter.name!r} has no definition")
                definitions[id(parameter.temp)] = argument
            else:
                slots[parameter.slot.id] = argument

        def write(ref, value):
            if self.ssa:
                _assert(isinstance(ref, Temp) and not isinstance(ref, Slot),
                        f"{fn.name}: non-Temp definition {ref!r} in SSA IR")
                definitions[id(ref)] = value
            elif isinstance(ref, Slot):
                slots[ref.id] = value
            else:
                temps[id(ref)] = value

        def read(ref):
            if self.ssa:
                _assert(isinstance(ref, Temp),
                        f"{fn.name}: SSA read of non-Temp value {ref!r}")
                _assert(id(ref) in definitions,
                        f"{fn.name}: read of undefined SSA value {ref}")
                return definitions[id(ref)]
            if isinstance(ref, Slot):
                _assert(ref.id in slots,
                        f"{fn.name}: read of undefined slot {ref}")
                return slots[ref.id]
            _assert(id(ref) in temps,
                    f"{fn.name}: read of undefined temporary {ref}")
            return temps[id(ref)]

        def execute(instruction, block_label):
            if isinstance(instruction, Const):
                write(instruction.dest, instruction.value)
            elif isinstance(instruction, Copy):
                write(instruction.dest, read(instruction.src))
            elif isinstance(instruction, BinOp):
                left = read(instruction.left)
                right = read(instruction.right)
                write(instruction.dest,
                      self._apply_binop(instruction, left, right,
                                        fn.name, block_label))
            elif isinstance(instruction, Call):
                callee = self._functions.get(instruction.name)
                _assert(callee is not None,
                        f"{fn.name}: call to unknown function "
                        f"{instruction.name!r}")
                values = tuple(read(operand) for operand in instruction.args)
                _assert(
                    len(values) == len(callee.params),
                    f"call to {instruction.name!r}: arity mismatch at run "
                    f"time ({len(values)} for {len(callee.params)})",
                )
                trace.append((instruction.name, values))
                write(instruction.dest, self._invoke(callee, values, trace))
            else:
                _assert(False,
                        f"{fn.name}: unknown instruction {instruction!r}")

        known_blocks = {id(block) for block in fn.blocks}
        block = fn.entry
        previous = None
        steps_left = _STEP_LIMIT

        while True:
            _assert(steps_left > 0,
                    f"{fn.name}: step budget exhausted "
                    "(program does not terminate?)")
            steps_left -= 1

            if self.ssa:
                pending = []
                for phi in block.phis:
                    _assert(previous is not None,
                            f"{fn.name}:{block.label}: phi in entry block "
                            "has no predecessor edge")
                    source = phi.entries.get(previous)
                    _assert(source is not None,
                            f"{fn.name}:{block.label}: phi {phi.dest} has "
                            f"no incoming value from predecessor "
                            f"{previous.label}")
                    pending.append((phi.dest, read(source)))
                for destination, value in pending:
                    definitions[id(destination)] = value
            else:
                _assert(not block.phis,
                        f"{fn.name}:{block.label}: phi node in non-SSA IR")

            for instruction in block.instructions:
                execute(instruction, block.label)

            terminator = block.terminator
            _assert(terminator is not None,
                    f"{fn.name}:{block.label}: missing terminator")

            if isinstance(terminator, Return):
                if terminator.value is None:
                    return None
                return read(terminator.value)

            if isinstance(terminator, Jump):
                _assert(id(terminator.target) in known_blocks,
                        f"{fn.name}:{block.label}: jump to unknown block "
                        f"{terminator.target.label}")
                previous, block = block, terminator.target
                continue

            if isinstance(terminator, Branch):
                condition = read(terminator.cond)
                _assert(isinstance(condition, bool),
                        f"{fn.name}:{block.label}: branch on non-bool "
                        f"value {condition!r}")
                target = (terminator.true_target if condition
                          else terminator.false_target)
                _assert(id(target) in known_blocks,
                        f"{fn.name}:{block.label}: branch to unknown block "
                        f"{target.label}")
                previous, block = block, target
                continue

            _assert(False,
                    f"{fn.name}:{block.label}: unknown terminator "
                    f"{terminator!r}")

    @staticmethod
    def _apply_binop(instruction, left, right, fn_name, block_label):
        if instruction.kind == "arith":
            _assert(type(left) is int and type(right) is int,
                    f"{fn_name}: arithmetic {instruction.operator!r} on "
                    f"non-int operands {left!r}, {right!r}")
            if instruction.operator in ("div", "mod"):
                if right == 0:
                    raise _Trap("div-by-zero", fn_name, block_label)
                if instruction.operator == "div":
                    return _trunc_div(left, right)
                return _trunc_mod(left, right)
            operation = _ARITH.get(instruction.operator)
            _assert(operation is not None,
                    f"{fn_name}: unsupported arithmetic operator "
                    f"{instruction.operator!r}")
            return operation(left, right)

        if instruction.kind == "compare":
            _assert(type(left) == type(right),
                    f"{fn_name}: comparison {instruction.operator!r} "
                    f"between mismatched value types {left!r}, {right!r}")
            operation = _COMPARE.get(instruction.operator)
            _assert(operation is not None,
                    f"{fn_name}: unsupported comparison operator "
                    f"{instruction.operator!r}")
            return operation(left, right)

        _assert(False, f"{fn_name}: unknown BinOp kind {instruction.kind!r}")


def _observe(module, entry, arguments):
    """Run ``entry`` and return the complete observable result.

    Normal termination: ``("return", value, trace)``.  A defined runtime
    error: ``("trap", category, function, block, trace-so-far)`` -- the
    trace of calls up to the trap is observable output and compared too.
    """
    trace = []
    try:
        value = _Interpreter(module).run(entry, arguments, trace)
    except _Trap as trap:
        return ("trap", trap.category, trap.function, trap.block,
                tuple(trace))
    return ("return", value, tuple(trace))


# --------------------------------------------------------------------------
# Pass orderings
# --------------------------------------------------------------------------

_PASSES = {
    "ssa": to_ssa,
    "dce": eliminate_dead_code,
}

#: The default optimized pipeline (front end, then these passes, then
#: rendering).  Every variant below keeps SSA construction before the
#: SSA-dependent DCE pass; rendering is always the final stage.
_DEFAULT_ORDER = ("ssa", "dce")

#: (order name, pass steps applied after lowering) -- all legal variants.
_ORDERS = [
    ("lower -> ssa -> dce -> render (default)", _DEFAULT_ORDER),
    ("lower -> ssa -> render", ("ssa",)),
    ("lower -> ssa -> dce -> dce -> render", ("ssa", "dce", "dce")),
    ("lower -> ssa -> ssa -> dce -> render", ("ssa", "ssa", "dce")),
    ("lower -> ssa -> dce -> ssa -> dce -> render",
     ("ssa", "dce", "ssa", "dce")),
]


def _apply(module, steps):
    """Apply the named passes to ``module`` in order."""
    for step in steps:
        module = _PASSES[step](module)
    return module


def _compile(ast, steps):
    """Compile ``ast`` through lowering plus ``steps`` (rendering excluded)."""
    return _apply(lower_module(ast), steps)


# --------------------------------------------------------------------------
# Test programs: shapes where passes genuinely interact
# --------------------------------------------------------------------------


def _merge_consts_ast():
    # Constants and phi propagation at a branch merge: `v` takes a
    # different constant expression on each edge and the merged value is
    # consumed after the join.
    return program(
        func("merge_consts", [param("c", "bool")], "int", [
            let("v", "int", int_(0)),
            if_(
                var("c"),
                [assign("v", arith("mul", int_(2), int_(3)))],
                [assign("v", int_(10))],
            ),
            let("w", "int", arith("add", var("v"), int_(1))),
            ret(var("w")),
        ]),
    )


def _dead_vs_calls_ast():
    # Removable pure computations next to side-effecting calls: the dead
    # chains (a/b, s) may vanish, but both calls and their order are
    # observable and must survive every order.
    return program(
        func("side", [param("n", "int")], "int",
             [ret(arith("add", var("n"), int_(1)))]),
        func("dead_vs_calls", [param("n", "int")], "int", [
            let("a", "int", arith("mul", var("n"), int_(2))),
            let("b", "int", arith("add", var("a"), int_(3))),
            let("r", "int", call("side", [var("n")])),
            let("s", "int", arith("mul", var("r"), var("r"))),
            let("t", "int", call("side", [arith("add", var("n"), int_(1))])),
            ret(var("r")),
        ]),
    )


def _loop_invariant_ast():
    # A loop-invariant computation (inv * 2) recomputed inside the body,
    # plus a conditional branch inside the loop selecting between two
    # updates of the carried value.
    return program(
        func("loop_invariant", [param("n", "int")], "int", [
            let("total", "int", int_(0)),
            let("inv", "int", arith("mul", int_(7), int_(3))),
            let("i", "int", var("n")),
            while_(
                compare("gt", var("i"), int_(0)),
                [
                    let("k", "int", arith("mul", var("inv"), int_(2))),
                    if_(
                        compare("eq", arith("mod", var("i"), int_(2)),
                                int_(0)),
                        [assign("total",
                                arith("add", var("total"), var("k")))],
                        [assign("total",
                                arith("add", var("total"), int_(1)))],
                    ),
                    assign("i", arith("sub", var("i"), int_(1))),
                ],
            ),
            ret(var("total")),
        ]),
    )


def _post_call_dead_ast():
    # An inlinable-shape call followed by dead code computed from its
    # result; only the live subtraction may keep the call result's chain.
    return program(
        func("dbl", [param("x", "int")], "int",
             [ret(arith("mul", var("x"), int_(2)))]),
        func("post_call_dead", [param("n", "int")], "int", [
            let("r", "int", call("dbl", [var("n")])),
            let("dead1", "int", arith("add", var("r"), int_(100))),
            let("dead2", "int", arith("mul", var("dead1"), var("dead1"))),
            let("out", "int", arith("sub", var("r"), int_(1))),
            ret(var("out")),
        ]),
    )


def _div_trap_ast():
    # Division by a computed value: traps with the language's defined
    # div-by-zero error exactly when the divisor is zero.
    return program(
        func("div_trap", [param("n", "int")], "int", [
            let("d", "int", arith("sub", var("n"), int_(3))),
            let("q", "int", arith("div", int_(100), var("d"))),
            ret(var("q")),
        ]),
    )


def _guarded_div_ast():
    # Division on a conditional path only: the zero-divisor input takes
    # the other branch and must terminate normally under every order.
    return program(
        func("guarded_div", [param("n", "int")], "int", [
            let("r", "int", int_(0)),
            if_(
                compare("ne", var("n"), int_(0)),
                [assign("r", arith("div", int_(10), var("n")))],
                [assign("r", int_(7))],
            ),
            ret(arith("add", var("r"), arith("mod", var("n"), int_(2)))),
        ]),
    )


def _trap_in_branch_ast():
    # The trap fires inside the then-block of an if, so the trap location
    # (function, block label) is a non-entry block -- a location every
    # legal order must reproduce exactly.
    return program(
        func("trap_in_branch", [param("n", "int")], "int", [
            let("r", "int", int_(0)),
            if_(
                compare("ge", var("n"), int_(10)),
                [assign("r", arith(
                    "div", var("n"), arith("sub", var("n"), int_(10))))],
                [assign("r", var("n"))],
            ),
            ret(var("r")),
        ]),
    )


# (case name, AST, entry function, [(arguments, expected observable)]).
# Expected values pin the ground truth independently of any pipeline.
_CASES = [
    (
        "branch-merge constants",
        _merge_consts_ast(), "merge_consts",
        [
            ((True,), ("return", 7, ())),
            ((False,), ("return", 11, ())),
        ],
    ),
    (
        "dead computations vs side-effecting calls",
        _dead_vs_calls_ast(), "dead_vs_calls",
        [
            ((3,), ("return", 4, (("side", (3,)), ("side", (4,))))),
            ((-1,), ("return", 0, (("side", (-1,)), ("side", (0,))))),
        ],
    ),
    (
        "loop with invariant and inner branch",
        _loop_invariant_ast(), "loop_invariant",
        [
            ((0,), ("return", 0, ())),
            ((2,), ("return", 43, ())),
            ((3,), ("return", 44, ())),
            ((4,), ("return", 86, ())),
        ],
    ),
    (
        "inlinable call with dead code after it",
        _post_call_dead_ast(), "post_call_dead",
        [
            ((5,), ("return", 9, (("dbl", (5,)),))),
            ((-3,), ("return", -7, (("dbl", (-3,)),))),
        ],
    ),
    (
        "division by computed value",
        _div_trap_ast(), "div_trap",
        [
            ((5,), ("return", 50, ())),
            ((0,), ("return", -33, ())),
            ((3,), ("trap", "div-by-zero", "div_trap", "b0", ())),
        ],
    ),
    (
        "guarded division on a conditional path",
        _guarded_div_ast(), "guarded_div",
        [
            ((0,), ("return", 7, ())),
            ((4,), ("return", 2, ())),
            ((3,), ("return", 4, ())),
            ((-3,), ("return", -4, ())),
        ],
    ),
    (
        "trap inside a branch block",
        _trap_in_branch_ast(), "trap_in_branch",
        [
            ((5,), ("return", 5, ())),
            ((12,), ("return", 6, ())),
            ((10,), ("trap", "div-by-zero", "trap_in_branch", "b1", ())),
        ],
    ),
]


def _mismatch_message(case, order, arguments, baseline, variant, ast):
    return (
        f"first observable divergence for source sample {case!r} under "
        f"pass order {order!r} at arguments {arguments!r}\n"
        f"baseline (unoptimized) result: {baseline!r}\n"
        f"variant result:                {variant!r}\n"
        f"reproducible AST:\n"
        f"{json.dumps(ast, indent=2, ensure_ascii=False)}"
    )


# --------------------------------------------------------------------------
# Cross-order semantic equivalence
# --------------------------------------------------------------------------


class CrossOrderSemanticsTests(unittest.TestCase):
    def test_baseline_matches_ground_truth(self):
        # The unoptimized (lower-only) module is the semantic baseline;
        # pin it against the literal expected observables first.
        for name, ast, entry, runs in _CASES:
            for arguments, expected in runs:
                with self.subTest(case=name, arguments=arguments):
                    baseline = _observe(_compile(ast, ()), entry, arguments)
                    self.assertEqual(
                        baseline, expected,
                        msg=(
                            f"baseline mismatch for source sample {name!r} "
                            f"at arguments {arguments!r}: "
                            f"expected {expected!r}, got {baseline!r}"
                        ),
                    )

    def test_every_legal_order_preserves_observable_semantics(self):
        for name, ast, entry, runs in _CASES:
            baseline_module = _compile(ast, ())
            for order_name, steps in _ORDERS:
                variant_module = _compile(ast, steps)
                for arguments, _expected in runs:
                    with self.subTest(case=name, order=order_name,
                                      arguments=arguments):
                        baseline = _observe(
                            baseline_module, entry, arguments)
                        variant = _observe(
                            variant_module, entry, arguments)
                        # Return value, call trace, and -- on a defined
                        # runtime error -- error category and trap
                        # location must all be identical.
                        self.assertEqual(
                            baseline, variant,
                            msg=_mismatch_message(
                                name, order_name, arguments,
                                baseline, variant, ast),
                        )

    def test_text_equality_is_not_the_cross_order_criterion(self):
        # Pin the premise of the suite: legal orders genuinely produce
        # different emitted text (DCE removes the dead chains), so the
        # semantic comparison above is doing real work -- correctness
        # across orders is never judged by target-text equality.
        ast = _dead_vs_calls_ast()
        text_without_dce = render_module(_compile(ast, ("ssa",)))
        text_with_dce = render_module(_compile(ast, ("ssa", "dce")))
        self.assertNotEqual(text_without_dce, text_with_dce)
        for arguments in ((3,), (-1,)):
            self.assertEqual(
                _observe(_compile(ast, ("ssa",)), "dead_vs_calls",
                         arguments),
                _observe(_compile(ast, ("ssa", "dce")), "dead_vs_calls",
                         arguments),
            )


# --------------------------------------------------------------------------
# Byte-identity of repeated compilation
# --------------------------------------------------------------------------


class ByteIdentityTests(unittest.TestCase):
    def test_same_order_repeated_compiles_are_byte_identical(self):
        orders = [("lower -> render (unoptimized)", ())] + _ORDERS
        for name, ast, _entry, _runs in _CASES:
            for order_name, steps in orders:
                with self.subTest(case=name, order=order_name):
                    texts = [render_module(_compile(ast, steps))
                             for _ in range(3)]
                    self.assertEqual(
                        texts[0], texts[1],
                        msg=(f"non-deterministic output for source sample "
                             f"{name!r} under pass order {order_name!r} "
                             f"(compile 1 vs 2)"),
                    )
                    self.assertEqual(
                        texts[1], texts[2],
                        msg=(f"non-deterministic output for source sample "
                             f"{name!r} under pass order {order_name!r} "
                             f"(compile 2 vs 3)"),
                    )

    def test_default_pipeline_is_byte_identical_across_processes(self):
        # Fresh interpreters with different hash seeds must emit the same
        # bytes as this process for the default pipeline.
        repo_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        driver = (
            "import json, sys\n"
            "from compiler_ir import (\n"
            "    eliminate_dead_code, lower_module, render_module, to_ssa,\n"
            ")\n"
            "ast = json.load(sys.stdin)\n"
            "module = eliminate_dead_code(to_ssa(lower_module(ast)))\n"
            "sys.stdout.write(render_module(module))\n"
        )
        for name, ast, _entry, _runs in _CASES:
            expected = render_module(_compile(ast, _DEFAULT_ORDER))
            for seed in ("0", "1"):
                with self.subTest(case=name, hash_seed=seed):
                    env = dict(os.environ, PYTHONHASHSEED=seed)
                    proc = subprocess.run(
                        [sys.executable, "-c", driver],
                        input=json.dumps(ast),
                        capture_output=True, text=True,
                        cwd=repo_root, env=env, timeout=60,
                    )
                    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                    self.assertEqual(
                        proc.stdout, expected,
                        msg=(f"cross-process divergence for source sample "
                             f"{name!r} with PYTHONHASHSEED={seed}"),
                    )


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def _definition_count(module):
    """Total number of value definitions (params, phis, instructions)."""
    count = 0
    for fn in module.functions:
        count += len(fn.params)
        for block in fn.blocks:
            count += len(block.phis) + len(block.instructions)
    return count


class IdempotencyTests(unittest.TestCase):
    def test_ssa_conversion_is_idempotent(self):
        for name, ast, _entry, _runs in _CASES:
            with self.subTest(case=name):
                ssa_once = to_ssa(lower_module(ast))
                ssa_twice = to_ssa(ssa_once)
                self.assertEqual(
                    render_module(ssa_once), render_module(ssa_twice),
                    msg=(f"to_ssa is not idempotent for source sample "
                         f"{name!r}: the normalized IR changed on the "
                         f"second application"),
                )

    def test_dce_is_idempotent(self):
        for name, ast, _entry, _runs in _CASES:
            with self.subTest(case=name):
                once = eliminate_dead_code(to_ssa(lower_module(ast)))
                twice = eliminate_dead_code(once)
                # A second application performs no new removals...
                self.assertEqual(
                    _definition_count(once), _definition_count(twice),
                    msg=(f"eliminate_dead_code removed additional "
                         f"definitions on its second application for "
                         f"source sample {name!r}"),
                )
                # ...and the emitted text is unchanged.
                self.assertEqual(
                    render_module(once), render_module(twice),
                    msg=(f"eliminate_dead_code is not idempotent for "
                         f"source sample {name!r}: the emitted text "
                         f"changed on the second application"),
                )

    def test_full_sequence_reapplication_is_stable(self):
        # Re-applying the default sequence reaches a fixed point.  DCE
        # keeps the surviving SSA numbers (leaving holes), and the
        # re-normalizing to_ssa compacts them exactly once; from the
        # normalized module onward the sequence changes neither the
        # normalized IR nor the final emitted text.
        for name, ast, _entry, _runs in _CASES:
            with self.subTest(case=name):
                optimized = _compile(ast, _DEFAULT_ORDER)
                normalized = to_ssa(optimized)  # one-shot renumbering
                # DCE after renormalization removes nothing further...
                self.assertEqual(
                    render_module(normalized),
                    render_module(eliminate_dead_code(normalized)),
                    msg=(f"eliminate_dead_code changed the renormalized "
                         f"module for source sample {name!r}"),
                )
                # ...and the whole default sequence is a fixed point
                # from here on.
                self.assertEqual(
                    render_module(normalized),
                    render_module(_apply(normalized, _DEFAULT_ORDER)),
                    msg=(f"the default pass sequence is not a fixed "
                         f"point on its own normalized result for "
                         f"source sample {name!r}"),
                )


# --------------------------------------------------------------------------
# Stage constraints and front-end diagnostics
# --------------------------------------------------------------------------


class OrderingConstraintTests(unittest.TestCase):
    def test_dce_before_ssa_violates_the_stage_constraint(self):
        # SSA construction must precede the SSA-dependent optimization:
        # skipping to_ssa is rejected, which is what keeps every legal
        # ordering anchored on "ssa before dce".
        for name, ast, _entry, _runs in _CASES:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    eliminate_dead_code(lower_module(ast))

    def test_frontend_diagnostics_do_not_depend_on_pass_order(self):
        # Type errors and undefined symbols are diagnosed by the shared
        # front end before any pass-order divergence; every legal order
        # must surface the identical error type for the same bad source.
        bad_programs = [
            ("return type mismatch",
             program(func("f", [], "int", [ret(bool_(True))])),
             TypeCheckError),
            ("arithmetic on bool",
             program(func("f", [], "int",
                          [ret(arith("add", bool_(True), int_(1)))])),
             TypeCheckError),
            ("undefined variable",
             program(func("f", [], "int", [ret(var("ghost"))])),
             UndefinedSymbolError),
            ("undefined callee",
             program(func("f", [], "int", [ret(call("ghost", []))])),
             UndefinedSymbolError),
        ]
        orders = [("lower", ())] + _ORDERS
        for label, ast, error_type in bad_programs:
            for order_name, steps in orders:
                with self.subTest(diagnostic=label, order=order_name):
                    with self.assertRaises(error_type):
                        _compile(ast, steps)


if __name__ == "__main__":
    unittest.main()
