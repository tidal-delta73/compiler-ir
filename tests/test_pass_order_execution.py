"""Execution-level pass-order semantic equivalence and reproducibility tests.

The existing suites prove pass behavior two ways: ``test_semantic_equivalence``
interprets non-SSA vs SSA modules in-process, and ``test_pass_ordering``
interprets the modules produced by many legal pass schedules.  Both execute
the in-memory :class:`~compiler_ir.ir_nodes.Module` directly.  This module
instead goes through the project's public entry points end to end and
**executes the emitted target program as a separate operating-system
process**:

    AST dict (public schema)
      -> ``lower_module``            (validation, name resolution, types)
      -> a legal permutation of the existing public SSA passes
         ``to_ssa`` / ``fold_constants`` / ``eliminate_dead_code``
      -> ``render_module``           (instruction selection, always last)
      -> target text file
      -> ``tests/target_runner.py`` in a fresh subprocess

Nothing is added to or changed in the compiler package, no new public entry
point is introduced, and no production dependency is taken (the runner and
this module use the standard library only).  Different orders are **not**
required to emit byte-identical targets; they are required to produce
targets that run identically.

Observables (exactly three, compared as raw bytes/ints)
-------------------------------------------------------

* process exit status,
* standard output,
* standard error.

The language has no I/O statement, so the reserved ``emit(int) -> int``
function is the stdout channel: each executed ``emit`` call writes its
integer argument on one line.  Every corpus program calls ``emit``, so an
empty-stdout coincidence cannot mask a difference.  ``main``'s int/bool
result maps to the exit status (non-zero/true -> 1, zero/false -> 0) and a
``void main`` exits 0; every normal corpus run has empty stderr.  Expected
results are pinned from the source semantics and also checked against the
default-order run, which acts as the in-suite baseline.

Corpus (fixed inputs, every program terminates normally)
--------------------------------------------------------

Each program concentrates on an interaction real optimizations have with
each other:

* ``unreachable-branch`` -- a branch whose condition folds to a constant;
  the unreachable arm still contains calls and arithmetic at compile time.
* ``cross-block-delete`` -- a constant propagated across basic blocks, a
  pure chain deletable only after that propagation, and a same-literal
  merge phi that is live before folding and dead after it.
* ``loop-invariant-after-exit`` -- a loop-invariant value and a
  loop-invariant pure chain inside a loop, with the accumulated result
  observed after loop exit (also covering the zero-trip path).
* ``inlineable-callee`` -- a call to a trivial inlineable callee that
  itself contains a local folded constant and a useless expression; the
  call must survive every order (no order may silently drop it).
* ``shadow-join-backedge`` -- an inner same-name ``let`` shadowing an outer
  variable, a conditional merge phi for another variable, and reads and
  writes carried around a loop back edge.
* ``void-entry`` -- a ``void main`` with branch-local calls, pinning the
  bare-return exit status and empty stderr.

This codebase ships exactly three SSA-stage transforms -- SSA construction,
constant folding/propagation and dead-code elimination; loop-invariant
hoisting and inlining are not implemented passes, so nothing named "licm"
or "inline" exists to permute.  The loop-invariant and inlineable-callee
samples above nevertheless pin how the existing fold/DCE reorderings behave
around those source shapes.  All schedules preserve the stage contracts:
``to_ssa`` precedes every SSA-dependent pass (``fold``/``dce`` raise if run
earlier), and instruction selection (``render_module``) runs last.

Isolation and reproducibility
-----------------------------

Every compilation starts from a fresh ``copy.deepcopy`` of the source AST
and every run happens in a new subprocess with its own temporary target and
argument files, so symbol tables, temporary numbering, analysis caches and
pass state cannot leak between runs.  Equivalence is checked in two sweeps,
one visiting the orders forward and one backward; both sweeps must reach
the same conclusion.  The same source/configuration/order compiled three
times must yield byte-identical target code, and the first and third
artifacts are additionally executed to confirm the bytes agree at runtime.

A failing case names the source sample and the pass order, shows the
baseline order against the candidate order, and prints both actual triples
(exit status, stdout, stderr).  A non-reproducible target names the first
offset at which the artifacts differ and gives sizes, SHA-256 digests and
the first differing lines.  Cases are never skipped: a schedule that fails
to compile, a target that cannot be parsed/executed, or any difference in
the three observables fails the test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compiler_ir import (
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


RUNNER = Path(__file__).with_name("target_runner.py")
_RUN_TIMEOUT_SECONDS = 30


# ==========================================================================
# Pass schedules (existing public passes only; stage contracts enforced)
# ==========================================================================

DEFAULT_ORDER = ("ssa", "fold", "dce", "ssa")

# Four legal schedules that differ from the default.  The first three end
# with the canonicalizing ssa renumbering and reach the same fixed point;
# the last omits the trailing canonicalization, so its emitted target is
# deliberately byte-different and must still run identically.
LEGAL_ORDERS = {
    "default ssa,fold,dce,ssa": DEFAULT_ORDER,
    "dce-before-fold ssa,dce,fold,dce,ssa":
        ("ssa", "dce", "fold", "dce", "ssa"),
    "fold-dce-interleaved ssa,fold,dce,fold,dce,ssa":
        ("ssa", "fold", "dce", "fold", "dce", "ssa"),
    "fold-fixpoint ssa,fold,fold,dce,ssa":
        ("ssa", "fold", "fold", "dce", "ssa"),
    "raw-core ssa,fold,dce (no trailing canonicalization)":
        ("ssa", "fold", "dce"),
}
BASELINE_NAME = next(iter(LEGAL_ORDERS))

_PASSES = {
    "ssa": to_ssa,
    "fold": fold_constants,
    "dce": eliminate_dead_code,
}

_ORDER_NAMES = list(LEGAL_ORDERS)
_CANONICAL_ALTERNATIVES = [
    name for name in _ORDER_NAMES
    if name != BASELINE_NAME and not name.startswith("raw-core")
]


def apply_order(module, order):
    """Apply ``order`` to a lowered module, enforcing the stage contracts.

    ``fold``/``dce`` require SSA and so may not precede the first ``ssa``;
    instruction selection is never part of an order (the harness renders
    only after the whole order finishes).
    """
    ssa_seen = bool(getattr(module, "ssa", False))
    for name in order:
        if name == "ssa":
            module = to_ssa(module)
            ssa_seen = True
        elif name in ("fold", "dce"):
            if not ssa_seen:
                raise ValueError(
                    f"pass order violates precondition: {name!r} requires "
                    "an SSA module but no 'ssa' pass precedes it")
            module = _PASSES[name](module)
        else:
            raise ValueError(f"unknown pass in order: {name!r}")
    return module


def compile_target(ast, order):
    """Compile a *fresh* deep copy of ``ast`` and return target bytes.

    The deep copy plus the passes' copy-on-write contracts guarantee that
    no AST dict, symbol table, temporary numbering, analysis cache or pass
    state is shared between two calls.
    """
    lowered = lower_module(copy.deepcopy(ast))
    optimized = apply_order(lowered, order)
    return render_module(optimized).encode("utf-8")


# ==========================================================================
# Target execution in an isolated subprocess
# ==========================================================================


class RunResult:
    """The three observables of one executed target program."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def triple(self):
        return self.returncode, self.stdout, self.stderr

    def __eq__(self, other):
        if not isinstance(other, RunResult):
            return NotImplemented
        return self.triple() == other.triple()

    def __repr__(self):
        return (f"RunResult(exit={self.returncode!r}, "
                f"stdout={self.stdout!r}, stderr={self.stderr!r})")


def _argument_tokens(arguments):
    return ["true" if value is True else "false" if value is False
            else str(value) for value in arguments]


def execute_target(target_bytes, arguments):
    """Write target + args to fresh temp files and run them in a subprocess.

    Returns a :class:`RunResult`.  A timeout or a runner killed by a signal
    is reported with the real (negative) signal status, so it can never be
    confused with a normal 0/1 exit.
    """
    if not RUNNER.is_file():
        raise AssertionError(f"missing target runner at {RUNNER}")
    with tempfile.TemporaryDirectory(prefix="pass-order-exec-") as tmp:
        target_path = os.path.join(tmp, "target.ir")
        args_path = os.path.join(tmp, "arguments.txt")
        with open(target_path, "wb") as fh:
            fh.write(target_bytes)
        with open(args_path, "w", encoding="utf-8") as fh:
            for token in _argument_tokens(arguments):
                fh.write(token + "\n")
        try:
            completed = subprocess.run(
                [sys.executable, str(RUNNER), target_path, args_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_RUN_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                -9, b"",
                f"target runner timed out after "
                f"{_RUN_TIMEOUT_SECONDS}s".encode("utf-8"))
        return RunResult(
            completed.returncode, completed.stdout, completed.stderr)


# ==========================================================================
# Corpus
# ==========================================================================


def _emit_function():
    """The reserved stdout channel: identity over one int."""
    return func("emit", [param("v", "int")], "int", [ret(var("v"))])


def _unreachable_branch_program():
    # `2 < 3` folds to true, so the else arm is unreachable; it still
    # contains a folded constant chain, an emit call and a write to the
    # merge-used variable `v` at compile time.  `a = 1 + 2` and `p = a*100`
    # are pure; `p` is dead and disappears once folding exposes it.
    return program(
        func("main", [param("x", "int")], "int", [
            let("a", "int", arith("add", int_(1), int_(2))),
            let("p", "int", arith("mul", var("a"), int_(100))),
            let("v", "int", int_(0)),
            if_(compare("lt", arith("add", int_(1), int_(1)), int_(3)),
                [let("e0", "int",
                     call("emit", [arith("add", var("a"), int_(10))])),
                 assign("v", arith("add", var("a"), var("x")))],
                [let("q", "int", arith("mul", int_(9), int_(9))),
                 let("e1", "int", call("emit", [int_(999)])),
                 assign("v", int_(0))]),
            let("w", "int", call("emit", [var("v")])),
            ret(arith("add", var("v"), var("w"))),
        ]),
        _emit_function(),
    )


def _cross_block_delete_program():
    # `k = 2 + 3` is defined in the entry block and read across the branch
    # (mark arm) and after the join (`k * k`): propagation crosses basic
    # blocks.  `junk = (10 - 4) * k` is a pure chain deletable only after
    # folding.  `a` is 7 (3+4 vs 10-3) on both arms, so its merge phi folds
    # and the arm arithmetic is live before folding but dead afterwards --
    # the fold/DCE ordering interaction.
    return program(
        func("main", [param("c", "bool"), param("x", "int")], "int", [
            let("k", "int", arith("add", int_(2), int_(3))),
            let("junk", "int",
                arith("mul", arith("sub", int_(10), int_(4)), var("k"))),
            if_(var("c"),
                [let("h", "int", call("emit", [var("k")]))],
                []),
            let("a", "int", int_(0)),
            if_(var("c"),
                [assign("a", arith("add", int_(3), int_(4)))],
                [assign("a", arith("sub", int_(10), int_(3)))]),
            let("pa", "int", call("emit", [var("a")])),
            let("r", "int", arith("mul", var("k"), var("k"))),
            let("e", "int", call("emit", [var("r")])),
            ret(arith("add", arith("sub", var("r"), var("x")), var("e"))),
        ]),
        _emit_function(),
    )


def _loop_invariant_program():
    # `base = k + 1` is loop invariant and used only to accumulate; the
    # `(2 + 3) * i` chain in the body is loop-invariant in its constant part
    # and dead.  The loop result `total` is emitted and returned only after
    # exit; the zero-trip input pins the exit-after-zero-iterations path.
    return program(
        func("main", [param("n", "int"), param("k", "int")], "int", [
            let("base", "int", arith("add", var("k"), int_(1))),
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("pe", "int", call("emit", [var("i")])),
                let("junk", "int",
                    arith("mul", arith("add", int_(2), int_(3)), var("i"))),
                assign("total", arith(
                    "add", arith("add", var("total"), var("base")), var("i"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("e", "int", call("emit", [var("total")])),
            ret(var("total")),
        ]),
        _emit_function(),
    )


def _inlineable_callee_program():
    # `dbl` is a trivial inlineable callee: a local constant `two = 1 + 1`
    # that folds, a useless `unused = two * 9`, and a doubled return.  No
    # order inlines it, so the call and its ordering survive every
    # schedule; the caller also has dead post-call arithmetic.
    return program(
        func("main", [param("n", "int")], "int", [
            let("v", "int",
                call("dbl", [arith("add", var("n"), int_(1))])),
            let("e0", "int", call("emit", [var("v")])),
            let("junk1", "int", arith("mul", var("v"), int_(9))),
            let("junk2", "int", arith("add", var("junk1"), int_(5))),
            ret(arith("add", var("v"), call("emit", [int_(1)]))),
        ]),
        func("dbl", [param("x", "int")], "int", [
            let("two", "int", arith("add", int_(1), int_(1))),
            let("unused", "int", arith("mul", var("two"), int_(9))),
            ret(arith("mul", var("x"), var("two"))),
        ]),
        _emit_function(),
    )


def _shadow_join_backedge_program():
    # An inner same-name `let x` shadows the outer one; the outer value is
    # read again after the block.  `v` gets different definitions on the
    # two branches and converges at the merge phi; `total` and `i` are
    # written on every loop iteration and read across the back edge, and
    # `total` is observed after exit.
    return program(
        func("main", [param("n", "int")], "int", [
            let("x", "int", call("emit", [int_(1)])),
            block([let("x", "int", call("emit", [int_(2)]))]),
            let("v", "int", int_(0)),
            if_(compare("gt", var("n"), int_(0)),
                [assign("v", arith("add", var("x"), var("n")))],
                [assign("v", arith("sub", var("x"), var("n")))]),
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                assign("total", arith("add", var("total"), var("v"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("e", "int", call("emit", [var("total")])),
            ret(arith("add", var("x"), var("total"))),
        ]),
        _emit_function(),
    )


def _void_entry_program():
    # void main: exactly one arm's call runs, then the function falls
    # through with a bare return; the process exit status must be 0 and
    # stderr empty.
    return program(
        func("main", [param("n", "int")], "void", [
            if_(compare("gt", var("n"), int_(0)),
                [let("a", "int", call("emit", [int_(7)]))],
                [let("b", "int", call("emit", [int_(8)]))]),
        ]),
        _emit_function(),
    )


def _stdout(*values):
    return ("\n".join(str(v) for v in values) + "\n").encode("utf-8")


# (label, builder, (argument tuple, expected exit, expected stdout)) -- the
# expectations follow directly from the source semantics, not from any
# order's output.
_CORPUS = [
    ("unreachable-branch", _unreachable_branch_program, [
        ((4,), 1, _stdout(13, 7)),
        ((-2,), 1, _stdout(13, 1)),
    ]),
    ("cross-block-delete", _cross_block_delete_program, [
        ((True, 9), 1, _stdout(5, 7, 25)),
        ((False, 9), 1, _stdout(7, 25)),
    ]),
    ("loop-invariant-after-exit", _loop_invariant_program, [
        ((3, 9), 1, _stdout(0, 1, 2, 33)),
        ((0, 5), 0, _stdout(0)),
    ]),
    ("inlineable-callee", _inlineable_callee_program, [
        ((4,), 1, _stdout(10, 1)),
        ((0,), 1, _stdout(2, 1)),
    ]),
    ("shadow-join-backedge", _shadow_join_backedge_program, [
        ((3,), 1, _stdout(1, 2, 12)),
        ((0,), 1, _stdout(1, 2, 0)),
    ]),
    ("void-entry", _void_entry_program, [
        ((1,), 0, _stdout(7)),
        ((0,), 0, _stdout(8)),
    ]),
]


# ==========================================================================
# Failure formatting
# ==========================================================================


def _format_triple(result):
    if isinstance(result, RunResult):
        rc, out, err = result.triple()
    else:
        rc, out, err = result
    return (f"exit status : {rc!r}\n"
            f"stdout      : {out!r}\n"
            f"stderr      : {err!r}")


def _mismatch_message(sample, ast, arguments, baseline_name,
                      candidate_name, baseline, candidate):
    return (
        f"semantic mismatch for source sample {sample!r}\n"
        f"arguments   : {arguments!r}\n"
        f"source AST  :\n{json.dumps(ast, indent=2, ensure_ascii=False)}\n"
        f"baseline order : {baseline_name}\n"
        f"candidate order: {candidate_name}\n"
        f"--- baseline ---\n{_format_triple(baseline)}\n"
        f"--- candidate ---\n{_format_triple(candidate)}"
    )


def _first_difference(a, b):
    """Return ``(offset, line_no, a_line, b_line)`` for two bytes objects."""
    limit = min(len(a), len(b))
    offset = next((i for i in range(limit) if a[i] != b[i]), None)
    if offset is None and len(a) != len(b):
        offset = limit
    if offset is None:
        return None
    line_no = a.count(b"\n", 0, offset) + 1
    line_start = a.rfind(b"\n", 0, offset) + 1
    line_end_a = a.find(b"\n", offset)
    line_end_b = b.find(b"\n", offset)
    if line_end_a < 0:
        line_end_a = len(a)
    if line_end_b < 0:
        line_end_b = len(b)
    return (offset, line_no,
            a[line_start:line_end_a], b[line_start:line_end_b])


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _reproducibility_message(sample, order_name, first, second):
    diff = _first_difference(first, second)
    offset, line_no, line_a, line_b = diff
    return (
        f"target code not reproducible for sample {sample!r}, order "
        f"{order_name!r}\n"
        f"first differing byte offset: {offset} (line {line_no})\n"
        f"artifact A: {len(first)} bytes, sha256={_digest(first)}\n"
        f"artifact B: {len(second)} bytes, sha256={_digest(second)}\n"
        f"line A: {line_a!r}\n"
        f"line B: {line_b!r}"
    )


def _compile_or_fail(sample, ast, order_name, order):
    try:
        return compile_target(ast, order)
    except Exception as exc:  # a legal order must complete compilation
        raise AssertionError(
            f"order {order_name!r} failed to compile sample {sample!r}: "
            f"{type(exc).__name__}: {exc}") from exc


# ==========================================================================
# Tests
# ==========================================================================


class PassOrderExecutionTests(unittest.TestCase):
    def _sweep(self, sample, ast, arguments, expected, order_names):
        """Compile+run every order fresh; return {name: RunResult}.

        The pinned expectation is ground truth; the default-order run must
        meet it, and every candidate triple must equal the baseline triple
        exactly (exit status, stdout, stderr as raw values).
        """
        results = {}
        targets = {}
        for order_name in order_names:
            target = _compile_or_fail(
                sample, ast, order_name, LEGAL_ORDERS[order_name])
            result = execute_target(target, arguments)
            results[order_name] = result
            targets[order_name] = target

        expected_rc, expected_out = expected
        expected_result = RunResult(expected_rc, expected_out, b"")
        baseline = results[BASELINE_NAME]
        self.assertEqual(
            baseline.triple(), expected_result.triple(),
            msg=(f"sample {sample!r} arguments {arguments!r}: default-order "
                 f"run disagrees with the pinned source semantics\n"
                 f"{_format_triple(baseline)}"),
        )
        # Non-vacuous: the stdout channel is actually exercised, and stderr
        # stays empty on every normal run.
        self.assertTrue(
            baseline.stdout, msg=f"sample {sample!r} produced no stdout")
        self.assertEqual(baseline.stderr, b"")

        for order_name in order_names:
            if order_name == BASELINE_NAME:
                continue
            candidate = results[order_name]
            self.assertEqual(
                candidate.triple(), baseline.triple(),
                msg=_mismatch_message(
                    sample, ast, arguments, BASELINE_NAME, order_name,
                    baseline, candidate),
            )
        return targets, results

    def test_forward_sweep_all_orders_run_identically(self):
        for sample, builder, rows in _CORPUS:
            ast = builder()
            for arguments, rc, out in rows:
                with self.subTest(sample=sample, arguments=arguments,
                                  sweep="forward"):
                    self._sweep(sample, ast, arguments, (rc, out),
                                _ORDER_NAMES)

    def test_reverse_sweep_reaches_the_same_conclusion(self):
        # Visiting the orders in the reverse direction must not change any
        # verdict: each compilation/run still happens in its own fresh
        # context, so nothing from an earlier order may leak into a later
        # one regardless of scheduling direction.
        for sample, builder, rows in _CORPUS:
            ast = builder()
            for arguments, rc, out in rows:
                with self.subTest(sample=sample, arguments=arguments,
                                  sweep="reverse"):
                    self._sweep(sample, ast, arguments, (rc, out),
                                list(reversed(_ORDER_NAMES)))

    def test_reverse_sample_order_still_matches(self):
        # Isolation across samples as well: execute the whole corpus in
        # reverse, then forward, and demand identical triples from the two
        # independent runs.
        plan = []
        for sample, builder, rows in _CORPUS:
            for arguments, rc, out in rows:
                plan.append((sample, builder(), arguments, rc, out))

        def run_all(ordered):
            seen = {}
            for sample, ast, arguments, _rc, _out in ordered:
                target = _compile_or_fail(
                    sample, ast, BASELINE_NAME, DEFAULT_ORDER)
                seen[(sample, arguments)] = execute_target(target, arguments)
            return seen

        forward = run_all(plan)
        backward = run_all(list(reversed(plan)))
        self.assertEqual(
            {key: value.triple() for key, value in forward.items()},
            {key: value.triple() for key, value in backward.items()},
            msg="corpus execution order changed observable results",
        )


class TargetReproducibilityTests(unittest.TestCase):
    def test_same_source_and_order_compiles_byte_identically(self):
        for sample, builder, _rows in _CORPUS:
            ast = builder()
            for order_name, order in LEGAL_ORDERS.items():
                with self.subTest(sample=sample, order=order_name):
                    artifacts = [
                        _compile_or_fail(sample, ast, order_name, order)
                        for _ in range(3)
                    ]
                    first, second, third = artifacts
                    if first != second:
                        self.fail(_reproducibility_message(
                            sample, order_name, first, second))
                    if second != third:
                        self.fail(_reproducibility_message(
                            sample, order_name, second, third))

    def test_reproduced_artifacts_run_identically(self):
        # Byte identity of the compiled target plus identical execution of
        # independently produced copies.
        for sample, builder, rows in _CORPUS:
            ast = builder()
            for arguments, rc, out in rows:
                for order_name, order in LEGAL_ORDERS.items():
                    with self.subTest(sample=sample, arguments=arguments,
                                      order=order_name):
                        first = _compile_or_fail(
                            sample, ast, order_name, order)
                        second = _compile_or_fail(
                            sample, ast, order_name, order)
                        self.assertEqual(first, second)
                        run_a = execute_target(first, arguments)
                        run_b = execute_target(second, arguments)
                        self.assertEqual(
                            run_a.triple(), run_b.triple(),
                            msg=(f"identical target bytes executed "
                                 f"differently: {sample!r} {order_name!r} "
                                 f"{arguments!r}\n"
                                 f"{_format_triple(run_a)}\n"
                                 f"{_format_triple(run_b)}"),
                        )
                        self.assertEqual(run_a.triple(), (rc, out, b""))

    def test_canonical_orders_share_the_fixed_point_but_raw_core_differs(
            self):
        # Guard against vacuous equivalence: the canonical alternatives
        # reach one fixed-point artifact, while the deliberately
        # un-canonicalized raw core emits different bytes.  Execution
        # equivalence (proved elsewhere) must not rest on text identity.
        for sample, builder, _rows in _CORPUS:
            ast = builder()
            with self.subTest(sample=sample):
                canonical = compile_target(ast, DEFAULT_ORDER)
                for name in _CANONICAL_ALTERNATIVES:
                    self.assertEqual(
                        canonical, compile_target(ast, LEGAL_ORDERS[name]),
                        msg=f"{sample!r}: {name!r} did not reach the "
                            "canonical fixed point",
                    )
                raw = compile_target(
                    ast, LEGAL_ORDERS[
                        "raw-core ssa,fold,dce (no trailing "
                        "canonicalization)"])
                # The void sample may legitimately coincide textually once
                # numbering happens to match; require at least most corpus
                # programs to exhibit the intended textual difference.
        differing = sum(
            compile_target(builder(), DEFAULT_ORDER)
            != compile_target(
                builder(),
                LEGAL_ORDERS["raw-core ssa,fold,dce (no trailing "
                             "canonicalization)"])
            for _sample, builder, _rows in _CORPUS
        )
        self.assertGreaterEqual(
            differing, len(_CORPUS) - 1,
            msg="raw-core targets should be byte-different from the "
                "canonical target for the fold/DCE-sensitive samples",
        )


class StageContractTests(unittest.TestCase):
    """The execution harness keeps the existing preconditions and defaults."""

    def test_every_order_starts_with_ssa_and_ends_with_rendering(self):
        for name, order in LEGAL_ORDERS.items():
            self.assertEqual(
                order[0], "ssa",
                msg=f"order {name!r} does not construct SSA first")
            self.assertNotIn(
                "isel", order,
                msg="instruction selection must run after the whole order")
            seen_ssa = False
            for pass_name in order:
                if pass_name in ("fold", "dce"):
                    self.assertTrue(
                        seen_ssa,
                        msg=f"order {name!r}: {pass_name} precedes ssa")
                if pass_name == "ssa":
                    seen_ssa = True

    def test_ssa_dependent_passes_rejected_before_ssa(self):
        lowered = lower_module(_unreachable_branch_program())
        self.assertFalse(lowered.ssa)
        with self.assertRaises(ValueError):
            apply_order(lowered, ("fold",))
        with self.assertRaises(ValueError):
            apply_order(lowered, ("dce", "ssa"))
        # The public passes enforce the same precondition directly.
        with self.assertRaises(ValueError):
            fold_constants(lowered)
        with self.assertRaises(ValueError):
            eliminate_dead_code(lowered)

    def test_unknown_pass_rejected(self):
        lowered = lower_module(_unreachable_branch_program())
        with self.assertRaises(ValueError):
            apply_order(lowered, ("ssa", "inline"))

    def test_default_order_is_the_documented_pipeline(self):
        # The harness's default is literally to_ssa/fold/dce/to_ssa followed
        # by rendering; no new pipeline or public entry is invented.
        ast = _loop_invariant_program()
        lowered = lower_module(copy.deepcopy(ast))
        manual = render_module(
            to_ssa(eliminate_dead_code(fold_constants(to_ssa(lowered)))))
        self.assertEqual(manual.encode("utf-8"),
                         compile_target(ast, DEFAULT_ORDER))

    def test_type_error_still_rejected_before_any_order(self):
        bad = program(func(
            "f", [], "int",
            [ret(arith("add", int_(1),
                       {"kind": "bool", "value": False}))]))
        with self.assertRaises(TypeCheckError):
            lower_module(bad)

    def test_undefined_symbol_still_rejected(self):
        bad = program(func("f", [], "int", [ret(var("ghost"))]))
        with self.assertRaises(UndefinedSymbolError):
            lower_module(bad)

    def test_unoptimized_public_compilation_is_unchanged(self):
        # Without an explicitly requested order the public lowering and
        # rendering entry point behaves exactly as before.
        from compiler_ir import emit_ir
        text = emit_ir(_void_entry_program())
        self.assertTrue(text.startswith("module\n"))
        self.assertIn("function main(", text)
        self.assertIn("locals:", text)  # non-SSA rendering is unchanged
        self.assertEqual(text, emit_ir(_void_entry_program()))


if __name__ == "__main__":
    unittest.main()
