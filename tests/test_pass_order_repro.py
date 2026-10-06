"""Pass-order semantic-equivalence and reproducibility regression tests.

This module supplements ``test_pass_ordering.py`` (which pins pass
interactions, idempotence and structural shapes) with a fixed,
re-executable regression harness built only on the project's existing
public entry points -- :func:`lower_module`, :func:`to_ssa`,
:func:`fold_constants`, :func:`eliminate_dead_code` and
:func:`render_module`.  No public compile entry is added or changed, and
different legal pass orders are *not* required to emit byte-identical
target code; they are required to agree on observable behavior.

Observable criterion (process-style triple)
-------------------------------------------

Each compiled target program is *executed* by the deterministic,
standard-library-only IR interpreter already established in
``test_pass_ordering.py``, and the run is reduced to the same three
observables a process would expose:

* **exit status** -- ``0`` for normal termination, ``1`` for a defined
  runtime fault (the language's only fault is a zero divisor);
* **standard output** -- the deterministic text rendering of the ordered
  call trace (the language's only side effect) plus the ``return`` line;
* **standard error** -- empty on normal termination, otherwise the fault
  category, numbering-independent trigger site and faulting operands.

Two compilations of the same source are semantically equivalent iff all
three observables are exactly equal.  The tests never compare IR text
across orders to decide correctness, and no case is ever skipped.

Reproducibility criterion
-------------------------

For one and the same source, configuration and pass order, compiling at
least twice in a row (three times here, each time in a fresh compilation
context) must produce byte-identical target code.  A mismatch is
reported with the first differing byte offset (and line) plus the SHA-256
digests and sizes of both artifacts.

Pass orders
-----------

The default optimization pipeline is ``ssa, fold, dce, ssa`` (SSA
construction, constant folding/propagation, dead-code elimination, then
a canonicalizing renumbering; instruction selection -- the deterministic
:func:`render_module` emission -- is fixed last).  This compiler's
permutable optimization passes are constant folding/propagation and
dead-code elimination; there is no separate loop-invariant-motion or
inlining pass to schedule, so the alternative legal orders permute the
relative positions of ``fold`` and ``dce`` (fold before/after DCE,
repeated fold, interleavings, DCE-only) while preserving the stage
contracts the instruction-selection stage requires: SSA construction
precedes every SSA-dependent optimization and rendering stays final.
The corpus deliberately contains loop-invariant computations and
trivially inlinable callees so that every order proves it leaves their
behavior untouched.  Four candidate orders, all different from the
default, are exercised.

Corpus
------

Every program uses fixed literal inputs, is guaranteed to terminate
normally, and its expected output is pinned directly from the source
semantics.  The five required interaction categories are covered and
tagged:

* ``constant-condition`` -- an unreachable branch formed by a constant
  condition (a foldable comparison selects the arm at compile time, but
  the branch and both arms survive in the IR);
* ``cross-block-dead`` -- a computation that becomes deletable only
  after a constant is propagated across basic-block boundaries;
* ``loop-invariant`` -- a loop carrying an invariant whose accumulated
  result is consumed after the loop exit;
* ``inlinable-callee`` -- calls to a trivially inlinable function that
  itself contains local constants and useless expressions;
* ``shadow-join-backedge`` -- variable reads/writes with same-name
  shadowing, a conditional join and a loop back edge.

Isolation
---------

Every compilation runs in an independent context (a fresh deep copy of
the source AST lowered by a fresh :func:`lower_module` call), and the
whole case x order matrix is executed in several different evaluation
sequences -- forward, reversed, rotated and transposed -- which must all
reach identical conclusions, proving that no symbol table, temporary
numbering, analysis cache or pass state leaks from one run into the
next.  An unrelated "disturber" compilation interleaved between two
compilations of the same source must not change the result either.

Existing behavior (type-error and undefined-symbol diagnostics, the
default optimization configuration and the result of compiling without
an explicitly specified order) is pinned as unchanged; this module adds
tests and test helpers only.
"""
import copy
import hashlib
import unittest

from compiler_ir import (
    TypeCheckError,
    UndefinedSymbolError,
    lower_module,
    render_module,
)

from test_pipeline import (
    arith,
    assign,
    block,
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
from test_pass_ordering import (
    DEFAULT_ORDER,
    _interpret,
    apply_order,
)


# ==========================================================================
# Pass schedules: the default pipeline plus four legal alternatives that
# permute the fold/dce positions (SSA construction first, rendering last).
# ==========================================================================

BASELINE_ORDER = DEFAULT_ORDER  # ("ssa", "fold", "dce", "ssa")
BASELINE_NAME = "default ssa,fold,dce,ssa"

CANDIDATE_ORDERS = {
    # DCE runs before folding; a second DCE reclaims what folding exposes.
    "alt ssa,dce,fold,dce,ssa": ("ssa", "dce", "fold", "dce", "ssa"),
    # Folding is repeated to its fixpoint before DCE runs at all.
    "alt ssa,fold,fold,dce,ssa": ("ssa", "fold", "fold", "dce", "ssa"),
    # Fold and DCE interleave; each pass sees the other's output.
    "alt ssa,fold,dce,fold,dce,ssa": ("ssa", "fold", "dce", "fold", "dce",
                                      "ssa"),
    # No folding at all: constants never propagate, nothing fold-dead is
    # created; text differs, observable behavior must not.
    "alt ssa,dce,ssa": ("ssa", "dce", "ssa"),
}

ALL_ORDERS = {BASELINE_NAME: BASELINE_ORDER, **CANDIDATE_ORDERS}


# ==========================================================================
# Compilation and execution harness (test helpers only)
# ==========================================================================


def compile_target(ast, order):
    """Compile a fresh copy of ``ast`` through lowering and ``order``.

    Returns ``(target_text, optimized_module)``; the target text is the
    instruction-selection output (rendering), always produced last.  Each
    call is an independent compilation context: a deep-copied AST and a
    fresh :func:`lower_module` run, so no symbol table, temporary
    numbering, analysis cache or pass state can leak between calls.
    """
    lowered = lower_module(copy.deepcopy(ast))
    optimized = apply_order(lowered, order)
    return render_module(optimized), optimized


def _render_outcome(outcome):
    """Reduce an interpreter ``Outcome`` to (exit status, stdout, stderr)."""
    lines = [
        "call {}({})".format(name, ", ".join(str(arg) for arg in args))
        for name, args in outcome.output
    ]
    if outcome.kind == "normal":
        lines.append(f"return {outcome.value}")
        return 0, "".join(line + "\n" for line in lines), ""
    stderr = (
        f"runtime error: {outcome.category} at {outcome.site} "
        f"operands {outcome.operands}\n"
    )
    return 1, "".join(line + "\n" for line in lines), stderr


def run_target(module, entry, arguments):
    """Execute a compiled target program; return the observable triple."""
    return _render_outcome(_interpret(module, entry, arguments))


def compile_and_run(ast, order, entry, arguments):
    """Compile in a fresh context and run; return (target text, triple)."""
    text, optimized = compile_target(ast, order)
    return text, run_target(optimized, entry, arguments)


# ==========================================================================
# Failure formatting
# ==========================================================================


def _format_triple(triple):
    status, stdout, stderr = triple
    return (
        f"exit status: {status}\n"
        f"standard output:\n{stdout if stdout else '<empty>'}\n"
        f"standard error:\n{stderr if stderr else '<empty>'}"
    )


def _mismatch_message(label, baseline_name, candidate_name, entry, arguments,
                      baseline_triple, candidate_triple):
    return (
        f"semantic mismatch for source case {label!r}\n"
        f"baseline order: {baseline_name!r}\n"
        f"candidate order: {candidate_name!r}\n"
        f"entry {entry!r}, arguments {arguments!r}\n"
        f"--- baseline run ---\n{_format_triple(baseline_triple)}\n"
        f"--- candidate run ---\n{_format_triple(candidate_triple)}"
    )


def _first_byte_difference(first: bytes, second: bytes):
    for offset, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return offset
    if len(first) != len(second):
        return min(len(first), len(second))
    return None


def _byte_mismatch_message(label, order_name, first: bytes, second: bytes):
    offset = _first_byte_difference(first, second)
    line = first[:offset].count(b"\n") + 1
    return (
        f"non-reproducible target code for source case {label!r}, "
        f"order {order_name!r}\n"
        f"first differing byte offset: {offset} (line {line})\n"
        f"first artifact:  sha256 {hashlib.sha256(first).hexdigest()} "
        f"({len(first)} bytes)\n"
        f"second artifact: sha256 {hashlib.sha256(second).hexdigest()} "
        f"({len(second)} bytes)"
    )


# ==========================================================================
# Test corpus: one builder per required interaction category.
# ==========================================================================


def _emit_function(name="emit"):
    return func(name, [param("v", "int")], "int", [ret(var("v"))])


# Category: constant-condition.  `(1 + 2) < 2` folds to false, so the then
# arm (and its call) is unreachable at run time; the else arm's pure `3 * 0`
# offset folds away.  No order may delete the branch or move the call.
def _constant_condition_program():
    return program(
        func("constbranch", [param("x", "int")], "int", [
            if_(compare("lt", arith("add", int_(1), int_(2)), int_(2)),
                [
                    let("e", "int",
                        call("emit", [arith("add", var("x"), int_(1000))])),
                    ret(var("e")),
                ],
                [
                    let("e2", "int", call("emit", [var("x")])),
                    ret(arith("add", var("e2"),
                              arith("mul", int_(3), int_(0)))),
                ]),
        ]),
        _emit_function(),
    )


# Category: cross-block-dead.  `k` folds to 20 in the entry block and is
# propagated into the branch arm (the call argument) and past the join (the
# returned sum); `dead` is a pure cross-block computation nobody consumes,
# reclaimed only once folding exposes it to DCE.
def _cross_block_dead_program():
    return program(
        func("crossdead", [param("c", "bool"), param("n", "int")], "int", [
            let("k", "int",
                arith("mul", arith("add", int_(2), int_(3)), int_(4))),
            if_(var("c"), [let("h", "int", call("mark", [var("k")]))], []),
            let("dead", "int",
                arith("sub", arith("mul", var("k"), var("k")), var("n"))),
            ret(arith("add", var("k"), var("n"))),
        ]),
        _emit_function("mark"),
    )


# Category: loop-invariant.  `base` is loop-invariant and defined before the
# header; `junk` is a pure per-iteration computation that is always dead;
# the accumulated loop result is consumed by the return after the exit.
def _loop_invariant_program():
    return program(
        func("invloop", [param("n", "int"), param("k", "int")], "int", [
            let("base", "int",
                arith("add", arith("mul", var("k"), int_(2)), int_(3))),
            let("acc", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("junk", "int",
                    arith("sub", arith("mul", var("base"), var("i")),
                          var("base"))),
                assign("acc", arith("add", var("acc"), var("base"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            ret(arith("add", arith("mul", var("acc"), int_(1)),
                      arith("sub", var("base"), var("base")))),
        ]),
    )


# Category: inlinable-callee.  `helper` is a trivial inline candidate whose
# body mixes a local constant (`lc = 4 + 6`) with a useless expression
# (`useless`); no pass order actually inlines it, so both calls and their
# order remain observable in every schedule.
def _inlinable_callee_program():
    return program(
        func("caller", [param("n", "int")], "int", [
            let("v", "int", call("helper", [arith("add", var("n"), int_(1))])),
            let("w", "int", call("helper", [int_(3)])),
            ret(arith("add", var("v"), var("w"))),
        ]),
        func("helper", [param("a", "int")], "int", [
            let("lc", "int", arith("add", int_(4), int_(6))),
            let("useless", "int",
                arith("sub", arith("mul", var("lc"), var("a")), var("a"))),
            ret(arith("add", arith("mul", var("a"), int_(2)), var("lc"))),
        ]),
    )


# Category: shadow-join-backedge.  The inner block's `x` shadows the outer
# one (its writes die with the block); the outer `x` is updated on both arms
# of a conditional join and carried around the loop back edge, then read
# after the exit.
def _shadow_join_backedge_program():
    return program(
        func("shadowed", [param("n", "int")], "int", [
            let("x", "int", int_(1)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                block([
                    let("x", "int", int_(10)),
                    assign("x", arith("add", var("x"), var("i"))),
                ]),
                if_(compare("eq", arith("mod", var("i"), int_(2)), int_(0)),
                    [assign("x", arith("add", var("x"), int_(100)))],
                    [assign("x", arith("add", var("x"), int_(0)))]),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("s", "int", call("emit", [var("x")])),
            ret(var("s")),
        ]),
        _emit_function(),
    )


# (label, category, AST builder, entry, arguments, pinned observable triple).
# The pin is ground truth derived by hand from the source semantics; the
# cross-order checks are layered on top of it.
_CASES = [
    (
        "constant condition makes the then arm unreachable",
        "constant-condition",
        _constant_condition_program, "constbranch", (7,),
        (0, "call emit(7)\nreturn 7\n", ""),
    ),
    (
        "constant condition with a negative input",
        "constant-condition",
        _constant_condition_program, "constbranch", (-4,),
        (0, "call emit(-4)\nreturn -4\n", ""),
    ),
    (
        "cross-block constant exposes a deletable computation",
        "cross-block-dead",
        _cross_block_dead_program, "crossdead", (True, 5),
        (0, "call mark(20)\nreturn 25\n", ""),
    ),
    (
        "cross-block constant with the call arm not taken",
        "cross-block-dead",
        _cross_block_dead_program, "crossdead", (False, -2),
        (0, "return 18\n", ""),
    ),
    (
        "loop invariant accumulated and used after the exit",
        "loop-invariant",
        _loop_invariant_program, "invloop", (4, 5),
        (0, "return 52\n", ""),
    ),
    (
        "loop invariant on the zero-trip path",
        "loop-invariant",
        _loop_invariant_program, "invloop", (0, 9),
        (0, "return 0\n", ""),
    ),
    (
        "inlinable callee with local constants and dead expressions",
        "inlinable-callee",
        _inlinable_callee_program, "caller", (4,),
        (0, "call helper(5)\ncall helper(3)\nreturn 36\n", ""),
    ),
    (
        "shadowing, conditional join and loop back edge",
        "shadow-join-backedge",
        _shadow_join_backedge_program, "shadowed", (3,),
        (0, "call emit(201)\nreturn 201\n", ""),
    ),
    (
        "shadowing and join on the zero-trip path",
        "shadow-join-backedge",
        _shadow_join_backedge_program, "shadowed", (0,),
        (0, "call emit(1)\nreturn 1\n", ""),
    ),
]

_REQUIRED_CATEGORIES = {
    "constant-condition",
    "cross-block-dead",
    "loop-invariant",
    "inlinable-callee",
    "shadow-join-backedge",
}


# ==========================================================================
# Tests
# ==========================================================================


class CorpusSanityTests(unittest.TestCase):
    """The harness itself: fixed corpus, distinct legal orders, no skips."""

    def test_corpus_covers_every_required_interaction_category(self):
        self.assertGreaterEqual(len(_CASES), 5)
        covered = {category for _label, category, *_rest in _CASES}
        self.assertEqual(covered, _REQUIRED_CATEGORIES)

    def test_at_least_three_candidate_orders_all_differ_from_default(self):
        self.assertGreaterEqual(len(CANDIDATE_ORDERS), 3)
        for name, order in CANDIDATE_ORDERS.items():
            with self.subTest(order=name):
                self.assertNotEqual(order, BASELINE_ORDER)
                # Stage contracts: SSA construction first, no SSA-dependent
                # pass before it, and no instruction-selection pass anywhere
                # (rendering is appended last by the harness).
                self.assertEqual(order[0], "ssa")
                for index, pass_name in enumerate(order):
                    if pass_name in ("fold", "dce"):
                        self.assertIn("ssa", order[:index])

    def test_baseline_is_the_default_optimization_configuration(self):
        # The baseline exercised here IS the project's default pipeline;
        # compiling without an explicitly specified order is unchanged.
        self.assertEqual(BASELINE_ORDER, ("ssa", "fold", "dce", "ssa"))
        self.assertEqual(BASELINE_ORDER, DEFAULT_ORDER)

    def test_every_order_compiles_every_case(self):
        # A legal order that cannot complete compilation must fail loudly
        # here, attributed to the source case and the order.
        for label, _category, builder, _entry, _args, _pin in _CASES:
            for order_name, order in ALL_ORDERS.items():
                with self.subTest(sample=label, order=order_name):
                    text, optimized = compile_target(builder(), order)
                    self.assertTrue(optimized.ssa)
                    self.assertIsInstance(text, str)
                    self.assertTrue(text.endswith("\n"))


class CrossOrderEquivalenceTests(unittest.TestCase):
    """Same source, different legal orders: identical observable triples."""

    def test_baseline_run_matches_the_pinned_expectation(self):
        for label, _cat, builder, entry, arguments, pinned in _CASES:
            with self.subTest(sample=label):
                _text, triple = compile_and_run(
                    builder(), BASELINE_ORDER, entry, arguments)
                self.assertEqual(
                    triple, pinned,
                    msg=(f"source case {label!r}: default pipeline disagrees "
                         f"with the pinned expectation\n"
                         f"--- pinned ---\n{_format_triple(pinned)}\n"
                         f"--- actual ---\n{_format_triple(triple)}"),
                )

    def test_every_candidate_order_matches_the_baseline(self):
        for label, _cat, builder, entry, arguments, pinned in _CASES:
            ast = builder()
            _baseline_text, baseline_triple = compile_and_run(
                ast, BASELINE_ORDER, entry, arguments)
            # Ground truth first: the baseline itself must meet the pin.
            self.assertEqual(baseline_triple, pinned,
                             msg=f"source case {label!r}: baseline broken")
            for candidate_name, candidate_order in CANDIDATE_ORDERS.items():
                with self.subTest(sample=label, order=candidate_name):
                    _text, candidate_triple = compile_and_run(
                        ast, candidate_order, entry, arguments)
                    self.assertEqual(
                        candidate_triple, baseline_triple,
                        msg=_mismatch_message(
                            label, BASELINE_NAME, candidate_name,
                            entry, arguments,
                            baseline_triple, candidate_triple),
                    )


class ReproducibilityTests(unittest.TestCase):
    """Same source, same configuration, same order: byte-identical target."""

    def test_repeated_compilation_is_byte_identical(self):
        for label, _cat, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            for order_name, order in ALL_ORDERS.items():
                with self.subTest(sample=label, order=order_name):
                    artifacts = [
                        compile_target(ast, order)[0].encode("utf-8")
                        for _ in range(3)
                    ]
                    for index in (1, 2):
                        self.assertEqual(
                            artifacts[0], artifacts[index],
                            msg=_byte_mismatch_message(
                                label, order_name,
                                artifacts[0], artifacts[index]),
                        )

    def test_repeated_runs_are_observably_identical(self):
        for label, _cat, builder, entry, arguments, _pin in _CASES:
            ast = builder()
            for order_name, order in ALL_ORDERS.items():
                with self.subTest(sample=label, order=order_name):
                    first = compile_and_run(ast, order, entry, arguments)[1]
                    second = compile_and_run(ast, order, entry, arguments)[1]
                    self.assertEqual(
                        first, second,
                        msg=(f"source case {label!r}, order {order_name!r}: "
                             f"non-deterministic run\n"
                             f"--- first ---\n{_format_triple(first)}\n"
                             f"--- second ---\n{_format_triple(second)}"),
                    )


class IsolationTests(unittest.TestCase):
    """Compilation contexts are independent of each other and of history."""

    def _run_matrix(self, order_sequence, orders_outer=False):
        """Compile and run the whole case x order matrix.

        ``order_sequence`` controls the order in which schedules are
        visited; ``orders_outer`` transposes the loops.  Returns a dict
        keyed by (case label, arguments, order name) with the full
        (target text, observable triple) result.
        """
        results = {}
        cases = [(label, builder, entry, arguments)
                 for label, _cat, builder, entry, arguments, _pin in _CASES]
        if orders_outer:
            for order_name, order in order_sequence:
                for label, builder, entry, arguments in cases:
                    results[(label, arguments, order_name)] = \
                        compile_and_run(builder(), order, entry, arguments)
        else:
            for label, builder, entry, arguments in cases:
                for order_name, order in order_sequence:
                    results[(label, arguments, order_name)] = \
                        compile_and_run(builder(), order, entry, arguments)
        return results

    def test_evaluation_sequence_does_not_change_any_conclusion(self):
        order_list = list(ALL_ORDERS.items())
        sequences = [
            order_list,                     # forward
            list(reversed(order_list)),     # reversed
            order_list[2:] + order_list[:2],  # rotated
        ]
        reference = self._run_matrix(sequences[0])
        for index, sequence in enumerate(sequences[1:], start=1):
            with self.subTest(sequence=index):
                self.assertEqual(self._run_matrix(sequence), reference)
        with self.subTest(sequence="transposed"):
            self.assertEqual(
                self._run_matrix(order_list, orders_outer=True), reference)

    def test_unrelated_compilation_between_repeats_has_no_effect(self):
        # Interleave a different, "state-polluting" program (many functions,
        # shadowed names, loops) between two compilations of the same source
        # and order: symbol tables, temporary numbering, analysis caches and
        # pass state must not leak across the boundary.
        shadow_program = _shadow_join_backedge_program()
        disturber = program(
            *shadow_program["functions"],  # shadowed + its emit callee
            _inlinable_callee_program()["functions"][1],
            _loop_invariant_program()["functions"][0],
        )
        for label, _cat, builder, entry, arguments, _pin in _CASES:
            ast = builder()
            for order_name, order in ALL_ORDERS.items():
                with self.subTest(sample=label, order=order_name):
                    first = compile_and_run(ast, order, entry, arguments)
                    # Pollute, then repeat in a fresh context.
                    compile_and_run(disturber, order, "shadowed", (3,))
                    second = compile_and_run(ast, order, entry, arguments)
                    self.assertEqual(
                        first, second,
                        msg=(f"source case {label!r}, order {order_name!r}: "
                             "an intervening unrelated compilation changed "
                             "the result (state leaked between compilation "
                             "contexts)"),
                    )


class ExistingBehaviorPreservedTests(unittest.TestCase):
    """Diagnostics and the default configuration are unchanged."""

    def test_type_error_still_raised_before_any_order(self):
        bad = program(func(
            "f", [], "int",
            [ret(arith("add", int_(1), {"kind": "bool", "value": False}))]))
        for order_name, order in ALL_ORDERS.items():
            with self.subTest(order=order_name):
                with self.assertRaises(TypeCheckError):
                    compile_target(bad, order)

    def test_undefined_symbol_still_raised_before_any_order(self):
        bad = program(func("f", [], "int", [ret(var("ghost"))]))
        for order_name, order in ALL_ORDERS.items():
            with self.subTest(order=order_name):
                with self.assertRaises(UndefinedSymbolError):
                    compile_target(bad, order)

    def test_default_pipeline_output_is_unchanged_by_this_harness(self):
        # Compiling through the default configuration twice -- once directly
        # through the public entry points, once through this harness -- must
        # give the same target text.
        ast = _loop_invariant_program()
        direct = render_module(apply_order(
            lower_module(copy.deepcopy(ast)), BASELINE_ORDER))
        via_harness, _module = compile_target(ast, BASELINE_ORDER)
        self.assertEqual(direct, via_harness)


if __name__ == "__main__":
    unittest.main()
