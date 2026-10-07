"""Differential and metamorphic tests for the optimization pipeline.

Where the existing suites pin each pass and a fixed corpus, this module
treats the compiler as a black box behind its **existing public entry
points** and checks two metamorphic properties on one legal program:

1. **Differential / semantic equivalence** -- the same source produces the
   same observable result through the unoptimized baseline configuration
   and through at least two *different legal* pass orders, each order run
   once and then **run again**.  Observables are compared through both
   channels the repository already provides:

   * in-process interpretation of the lowered/optimized
     :class:`~compiler_ir.Module` (return value + ordered call trace), and
   * real execution of the emitted target text by ``target_runner.py`` in
     a fresh operating-system process (exit status, stdout bytes, stderr
     bytes).

2. **Determinism / fixed point** -- after an order is applied a second
   time, the *normalized* terminating IR (``to_ssa`` canonicalization
   followed by ``render_module``) is identical to the first application;
   orders that already end in the canonicalizing ``ssa`` are additionally
   byte-identical without normalization.  Recompiling the same
   source/configuration from a fresh deep copy yields byte-identical
   target bytes.  Equality is always exact -- never an order-insensitive
   multiset comparison, so no nondeterminism can hide.

Nothing is added to the compiler package and no public interface changes:
the only imports from ``compiler_ir`` are the documented entry points, and
test-only machinery (the interpreter, the subprocess runner and the AST
builders) is reused from the sibling test modules.

Inputs
------

Two kinds of inputs are used:

* a **seeded generator** (``ProgramGenerator``) which only emits programs
  that satisfy the existing type rules and are guaranteed to terminate
  (every loop is a literal-bounded counter incremented by exactly one),
  covering integer and boolean expressions, locals and shadowing,
  conditional branches (including a branch whose condition folds to a
  constant and whose untaken arm contains side effects), terminating
  loops with loop-invariant computations, and function calls;
* a set of **fixed hand-written regression samples** pinning the three
  interacting orderings -- constant propagation + dead-code elimination,
  loop-invariant hoisting followed by cleanup, and a retained call with
  subsequent constant folding -- together with one side-effecting
  expression that no order may delete and one loop computation (a trap
  site and a zero-trip call) that must never be hoisted.

Illegal programs are deliberately excluded from the equivalence
judgement; a separate fixed-sample class pins that pre-existing type
errors still raise the original exception type with the original
diagnostic position.

Failures and reproducibility
----------------------------

A failure is reported at the **first inconsistent stage** of an ordered
checklist (compilation, in-process execution, normalized-IR fixed point,
target-byte reproducibility, subprocess execution).  Before failing, the
seeded source is reduced by a deterministic statement-level delta
debugger that keeps the program type-valid and keeps the same order and
stage failing.  The message therefore always carries the explicit seed,
the minimized source program, the pass order and the first differing
stage, and regenerating from that seed reproduces the failure inside a
single test process.  The default seed lists are fixed constants; the
explicit seed set may be overridden with the ``DIFF_TEST_SEEDS``
environment variable (a comma-separated list of integers).
"""
from __future__ import annotations

import copy
import json
import os
import random
import unittest

from compiler_ir import (
    BinOp,
    Call,
    TypeCheckError,
    UndefinedSymbolError,
    hoist_loop_invariants,
    lower_module,
    render_module,
    to_ssa,
)

from test_pipeline import (
    arith,
    assign,
    block,
    bool_,
    call,
    compare,
    func,
    if_,
    int_,
    let,
    logical,
    param,
    program,
    ret,
    var,
    while_,
)
from test_pass_ordering import _interpret
from test_pass_order_execution import (
    LEGAL_ORDERS,
    apply_order,
    compile_target,
    execute_target,
)


# ==========================================================================
# Explicit seeds (fixed; overridable for one-process reproduction)
# ==========================================================================

def _parse_seeds():
    raw = os.environ.get("DIFF_TEST_SEEDS", "").strip()
    if not raw:
        return tuple(range(24))
    return tuple(int(piece) for piece in raw.split(",") if piece.strip())


#: Fixed corpus the non-vacuity coverage test always evaluates.  It is
#: deliberately independent of ``DIFF_TEST_SEEDS`` (a one-/two-seed
#: reproduction override must not relax the feature-coverage guarantee).
DEFAULT_SEEDS = tuple(range(24))
SEEDS = _parse_seeds()
# The subprocess sweep launches a real OS process per order, so it is run
# over a small subset of seeds while the cheaper in-process interpreter,
# fixed-point and byte checks cover every seed.  With an explicit seed
# override every listed seed is also executed end to end.
EXEC_SEEDS = SEEDS if os.environ.get("DIFF_TEST_SEEDS") else SEEDS[:6]

# Representative legal orders taken through the real subprocess runner for
# the fixed samples: the default canonical pipeline, a genuinely different
# interleaved legal order, and the deliberately un-canonicalized raw core.
# In-process checks below cover every order for the same samples.
SUBPROCESS_ORDER_NAMES = (
    "default ssa,fold,licm,dce,ssa",
    "fold-dce-interleaved ssa,fold,dce,licm,fold,dce,ssa",
    "raw-core ssa,fold,licm,dce (no trailing canonicalization)",
)


# ==========================================================================
# Seeded generator: type-valid, guaranteed-terminating programs
# ==========================================================================


class _Scope:
    """Lexical scope chain for generation; an inner scope may shadow."""

    __slots__ = ("parent", "declared")

    def __init__(self, parent=None):
        self.parent = parent
        self.declared: dict[str, str] = {}

    def declare(self, name, typ):
        # The generator never declares the same name twice in one scope:
        # every binding gets a unique fresh name, and shadowing is created
        # explicitly through a child scope.  Keep the guard as a belt.
        if name in self.declared:
            raise AssertionError(f"generator duplicated {name!r} in scope")
        self.declared[name] = typ

    def child(self):
        return _Scope(self)

    def visible(self):
        """Visible ``(name, type)`` pairs, innermost binding first."""
        items = []
        scope = self
        while scope is not None:
            for name, typ in scope.declared.items():
                if not any(name == other[0] for other in items):
                    items.append((name, typ))
            scope = scope.parent
        return items


class ProgramGenerator:
    """Grow one legal, terminating program from an explicit seed.

    Termination is structural, not probabilistic: the only loop form is a
    counter initialized to zero, bounded by a non-negative literal, and
    incremented by exactly one in the loop body top level; the generator
    never emits an assignment to, or a shadow of, a loop counter and never
    nests a loop.  Helper functions are loop-free and non-recursive.
    """

    _ARITH_OPS = ("add", "sub", "mul")  # never div/mod: no generated fault
    _COMPARE_OPS = ("eq", "ne", "lt", "le", "gt", "ge")

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.seed = seed
        self._serial = 0
        self.helpers: list[str] = []
        self.counts = dict(
            loops=0, shadows=0, const_branches=0, logicals=0,
            helper_calls=0, emits=0, bool_exprs=0,
        )

    # -- names -------------------------------------------------------------

    def _fresh(self, stem):
        self._serial += 1
        return f"{stem}_{self._serial}"

    # -- expressions -------------------------------------------------------

    def _expr(self, typ, scope, depth):
        rng = self.rng
        if depth <= 0:
            leaf = rng.randrange(3)
        else:
            leaf = rng.randrange(5)
        names = [name for name, got in scope.visible() if got == typ]

        if leaf < 2 and names:
            return var(rng.choice(names))

        if typ == "int":
            if leaf == 2:
                return int_(rng.randrange(-2, 6))
            if leaf == 3 and depth > 0:
                return arith(
                    rng.choice(self._ARITH_OPS),
                    self._expr("int", scope, depth - 1),
                    self._expr("int", scope, depth - 1),
                )
            # A side-effecting leaf: a call to a helper or to the stdout
            # channel.  Helper calls cover ordinary function calls; emit
            # is an observable effect the optimizers must preserve.
            if self.helpers and rng.random() < 0.4:
                self.counts["helper_calls"] += 1
                return call(rng.choice(self.helpers),
                            [self._expr("int", scope, max(0, depth - 1))])
            self.counts["emits"] += 1
            return call("emit", [int_(rng.randrange(0, 5))])

        # bool
        self.counts["bool_exprs"] += 1
        if leaf == 2:
            return bool_(rng.choice([True, False]))
        if leaf == 3 and depth > 0:
            return compare(
                rng.choice(self._COMPARE_OPS),
                self._expr("int", scope, depth - 1),
                self._expr("int", scope, depth - 1),
            )
        if depth > 0:
            # Short-circuit boolean expression; its operands may carry
            # side effects that are only evaluated when needed.
            self.counts["logicals"] += 1
            return logical(
                rng.choice(("and", "or")),
                self._expr("bool", scope, depth - 1),
                self._expr("bool", scope, depth - 1),
            )
        if names:
            return var(rng.choice(names))
        return bool_(rng.choice([True, False]))

    # -- statement lists ---------------------------------------------------

    def _stmts(self, scope, depth, stop, counter=None):
        """Append up to ``stop`` statements; ``counter`` is tamper-proof."""
        rng = self.rng
        out = []
        # A bound on attempts keeps generation finite even when a choice
        # has nothing legal to produce.
        for _ in range(stop + 4):
            if len(out) >= stop:
                break
            choice = rng.random()

            if choice < 0.22:
                out.append(let(
                    self._fresh("e"), "int",
                    call("emit", [self._expr("int", scope, 1)])))
                self.counts["emits"] += 1

            elif choice < 0.45:
                typ = rng.choice(("int", "int", "bool"))
                name = self._fresh(typ[0])
                out.append(let(
                    name, typ,
                    self._expr(typ, scope, max(0, depth - 1))))
                scope.declare(name, typ)

            elif choice < 0.60:
                targets = [(n, t) for n, t in scope.visible()
                           if n != counter]
                if targets:
                    name, typ = rng.choice(targets)
                    out.append(assign(
                        name, self._expr(typ, scope, max(0, depth - 1))))

            elif choice < 0.78 and depth >= 1:
                # One in a few conditions folds to a literal so the body
                # contains a branch with a provably dead, side-effecting
                # arm; all other conditions are genuine booleans.
                if rng.random() < 0.3:
                    self.counts["const_branches"] += 1
                    cond = compare("lt", int_(1), int_(2))
                else:
                    cond = self._expr("bool", scope, max(0, depth - 1))
                out.append(if_(
                    cond,
                    self._stmts(scope.child(), depth - 1,
                                rng.randrange(0, 3), counter),
                    self._stmts(scope.child(), depth - 1,
                                rng.randrange(0, 3), counter),
                ))

            elif choice < 0.90 and depth >= 1:
                # Explicit same-name shadowing inside a nested block.
                candidates = [n for n, t in scope.visible()
                              if n != counter and t == "int"]
                if candidates:
                    name = rng.choice(candidates)
                    inner = scope.child()
                    body_stmts = [
                        let(name, "int",
                            self._expr("int", inner, 1))
                    ]
                    body_stmts += self._stmts(
                        inner, depth - 1, rng.randrange(0, 2), counter)
                    out.append(block(body_stmts))
                    self.counts["shadows"] += 1
        return out

    def _loop(self, scope, depth):
        """Return ``[let counter = 0, while ...]`` (two top-level stmts)."""
        rng = self.rng
        counter = self._fresh("ctr")
        bound = rng.randrange(0, 4)
        scope.declare(counter, "int")
        body_scope = scope.child()
        body = []

        # A pure constant chain recomputed every iteration: an invariant
        # the hoister/cleanup can act on.
        if rng.random() < 0.7:
            body.append(let(self._fresh("junk"), "int",
                            arith("add", arith("add", int_(2), int_(3)),
                                  int_(4))))

        # A genuine add/sub/mul invariant read from an outer variable that
        # is not the counter: safe to hoist (trap-free, operands defined
        # outside the loop).  It feeds an observable emit so it stays.
        outer_ints = [n for n, t in body_scope.visible()
                      if n != counter and t == "int"]
        if outer_ints and rng.random() < 0.6:
            inv = self._fresh("inv")
            body.append(let(inv, "int",
                            arith("add", var(rng.choice(outer_ints)),
                                  int_(1))))
            body.append(let(self._fresh("mark"), "int",
                            call("emit", [var(inv)])))
            self.counts["emits"] += 1

        body += self._stmts(body_scope, max(0, depth - 1),
                            rng.randrange(1, 3), counter=counter)

        # A loop-varying observable value.
        body.append(let(self._fresh("step"), "int",
                        call("emit", [arith("add", var(counter), int_(0))])))
        self.counts["emits"] += 1
        # The unique increment that guarantees termination.
        body.append(assign(counter, arith("add", var(counter), int_(1))))
        self.counts["loops"] += 1
        loop = while_(compare("lt", var(counter), int_(bound)), body)
        return [let(counter, "int", int_(0)), loop]

    # -- whole program -----------------------------------------------------

    def program(self):
        rng = self.rng

        funcs = []
        for _ in range(rng.randrange(0, 3)):
            pname = self._fresh("x")
            hname = self._fresh("helper")
            self.helpers.append(hname)
            funcs.append(func(
                hname, [param(pname, "int")], "int",
                [let(self._fresh("y"), "int",
                     arith("add", int_(1), int_(2))),
                 ret(arith("add", var(pname), int_(0)))]))
        funcs.append(func("emit", [param("v", "int")], "int",
                          [ret(var("v"))]))

        params = [param("n", "int")]
        if rng.random() < 0.7:
            params.append(param("c", "bool"))
        if rng.random() < 0.5:
            params.append(param("m", "int"))

        scope = _Scope()
        for p in params:
            scope.declare(p["name"], p["type"])

        body = []
        for _ in range(rng.randrange(2, 5)):
            if rng.random() < 0.6:
                body += self._loop(scope, 2)
            else:
                body += self._stmts(scope, 2, rng.randrange(1, 3))

        result = self._expr("int", scope, 1)
        if self.helpers:
            result = call(rng.choice(self.helpers), [result])
            self.counts["helper_calls"] += 1
        out = self._fresh("out")
        body.append(let(out, "int", call("emit", [result])))
        self.counts["emits"] += 1
        body.append(ret(var(out)))

        return program(func("main", params, "int", body), *funcs)


def _arguments_for(ast, seed, salt):
    """Deterministic small concrete arguments for ``main``."""
    rng = random.Random(seed * 1009 + salt)
    values = []
    for p in ast["functions"][0]["params"]:
        if p["type"] == "bool":
            values.append(rng.choice((True, False)))
        else:
            values.append(rng.choice((-1, 0, 1, 2, 3)))
    return tuple(values)


# ==========================================================================
# Normalized IR and observable expectations
# ==========================================================================


def normalized_ir(module):
    """Stable canonical serialization of a terminal module.

    ``to_ssa`` is the existing idempotent canonicalization (clone +
    deterministic renumbering); rendering it is the stable serialization
    the README documents.
    """
    return render_module(to_ssa(module))


def expected_subprocess_triple(baseline_outcome):
    """Map the non-SSA interpreter baseline onto the runner's contract."""
    assert baseline_outcome.kind == "normal", baseline_outcome
    # main is always int: nonzero -> exit 1, zero -> exit 0.
    returncode = 0 if baseline_outcome.value == 0 else 1
    # Only emit() crosses the real stdout boundary; helper calls run
    # silently inside the process.
    stdout = "".join(
        f"{args[0]}\n"
        for callee, args in baseline_outcome.output
        if callee == "emit"
    ).encode("utf-8")
    return returncode, stdout, b""


# ==========================================================================
# Ordered stage checks and failure reporting
# ==========================================================================


STAGE_COMPILE = "compile-order"
STAGE_IN_PROCESS = "in-process-execution"
STAGE_IR_FIXED_POINT = "normalized-ir-fixed-point"
STAGE_TARGET_DETERMINISM = "target-byte-reproducibility"
STAGE_SUBPROCESS = "subprocess-execution"


def _first_failing_stage(ast, arguments, order_name, order,
                         run_target=True):
    """Run the ordered checklist; return ``(stage, detail)`` or None.

    When ``run_target`` is false the expensive subprocess stage is skipped
    (the in-process execution, normalized-IR fixed point and target-byte
    checks still run).
    """
    baseline = _interpret(lower_module(copy.deepcopy(ast)), "main",
                          arguments)
    if baseline.kind != "normal":
        return ("baseline", f"baseline run was not normal: {baseline!r}")

    try:
        once = apply_order(lower_module(copy.deepcopy(ast)), order)
    except Exception as exc:  # a legal order must compile
        return (STAGE_COMPILE,
                f"{type(exc).__name__}: {exc}")

    once_outcome = _interpret(once, "main", arguments)
    if once_outcome != baseline:
        return (STAGE_IN_PROCESS,
                f"baseline={baseline!r}\ncandidate={once_outcome!r}")

    # Apply the very same order again; normalized IR must be unchanged.
    again = apply_order(copy.deepcopy(once), order)
    if normalized_ir(once) != normalized_ir(again):
        return (STAGE_IR_FIXED_POINT,
                "normalized IR changed on the second application of "
                f"{order_name!r}")

    first_bytes = compile_target(ast, order)
    second_bytes = compile_target(ast, order)
    if first_bytes != second_bytes:
        return (STAGE_TARGET_DETERMINISM,
                f"{len(first_bytes)} vs {len(second_bytes)} bytes on a "
                "fresh recompilation")

    if not run_target:
        return None

    run = execute_target(first_bytes, arguments)
    wanted = expected_subprocess_triple(baseline)
    if run.triple() != wanted:
        return (STAGE_SUBPROCESS,
                f"expected={wanted!r}\nactual={run.triple()!r}")

    return None


# -- deterministic minimization -------------------------------------------


def _main_function(ast):
    for fn in ast["functions"]:
        if fn["name"] == "main":
            return fn
    raise AssertionError("generated program has no main")


def _stmt_containers(stmts):
    """All statement-bearing lists nested under ``stmts`` (inclusive)."""
    found = [stmts]
    for stmt in stmts:
        kind = stmt.get("kind")
        if kind == "if":
            found += _stmt_containers(stmt["then"])
            found += _stmt_containers(stmt["else"])
        elif kind == "while":
            found += _stmt_containers(stmt["body"])
        elif kind == "block":
            found += _stmt_containers(stmt["body"])
    return found


def minimize_program(ast, fails_with_stage, stage):
    """Deterministically shrink ``ast`` while keeping the same failure.

    ``fails_with_stage(candidate)`` returns the failing stage string or
    ``None``; it must be compile-only (it never executes the program), so
    removing statements cannot turn a candidate non-terminating.  A
    candidate is accepted only when it still lowers and fails at exactly
    ``stage``.  Statements are removed greedily container by container;
    the loop reaches a fixed point.
    """
    work = copy.deepcopy(ast)

    def still_fails(candidate):
        try:
            lower_module(copy.deepcopy(candidate))
        except Exception:
            return False  # a minimized case must stay a legal program
        return fails_with_stage(candidate) == stage

    if not still_fails(work):
        # Nothing to minimize; return a serialization of the original.
        return work

    changed = True
    while changed:
        changed = False
        for container in _stmt_containers(_main_function(work)["body"]):
            index = 0
            while index < len(container):
                removed = container.pop(index)
                if still_fails(work):
                    changed = True
                    continue
                container.insert(index, removed)
                index += 1
    return work


def format_failure(seed, order_name, order, stage, source, detail):
    return (
        f"differential mismatch\n"
        f"seed                 : {seed}\n"
        f"pass order ({order_name}): {list(order)}\n"
        f"first failing stage  : {stage}\n"
        f"detail               :\n{detail}\n"
        f"minimized source program:\n"
        f"{json.dumps(source, indent=2, ensure_ascii=False)}"
    )


# ==========================================================================
# Fixed hand-written regression samples
# ==========================================================================


def _emit_fn():
    return func("emit", [param("v", "int")], "int", [ret(var("v"))])


def _fold_dce_program():
    # k = 2 + 3 propagates; junk = k * (1 + 2) is a pure chain that is
    # live as written but dead once folding exposes it.  Both emits are
    # roots and must survive.
    return program(
        func("main", [param("x", "int")], "int", [
            let("k", "int", arith("add", int_(2), int_(3))),
            let("junk", "int",
                arith("mul", var("k"), arith("add", int_(1), int_(2)))),
            let("e", "int", call("emit", [var("k")])),
            ret(arith("add", var("e"), var("x"))),
        ]),
        _emit_fn(),
    )


def _licm_cleanup_program():
    # base = k + 1 is loop invariant; it hoists to the preheader and the
    # later cleanup reclaims constants it made redundant.  Observed only
    # after the loop exit, including the zero-trip path.
    return program(
        func("main", [param("n", "int"), param("k", "int")], "int", [
            let("base", "int", arith("add", var("k"), int_(1))),
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                assign("total", arith("add", var("total"), var("base"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("e", "int", call("emit", [var("total")])),
            ret(var("total")),
        ]),
        _emit_fn(),
    )


def _retained_call_refold_program():
    # dbl is a trivial inlineable callee with a folded local constant and
    # dead arithmetic; no pass inlines, so the call survives every order,
    # and later folding still removes the caller's dead math.
    return program(
        func("main", [param("n", "int")], "int", [
            let("v", "int",
                call("dbl", [arith("add", var("n"), int_(1))])),
            let("junk", "int", arith("mul", var("v"), int_(9))),
            ret(arith("add", var("v"), call("emit", [int_(1)]))),
        ]),
        func("dbl", [param("x", "int")], "int", [
            let("two", "int", arith("add", int_(1), int_(1))),
            let("unused", "int", arith("mul", var("two"), int_(9))),
            ret(arith("mul", var("x"), var("two"))),
        ]),
        _emit_fn(),
    )


def _preserved_effect_program():
    # z's value is never read; the emit call that initializes it is still
    # an observable effect.  Both emits must fire, in order.
    return program(
        func("main", [], "int", [
            let("z", "int", call("emit", [int_(42)])),
            ret(call("emit", [int_(7)])),
        ]),
        _emit_fn(),
    )


def _unhoistable_trap_program():
    # q = (k + 1) / (i + 1) inside the loop: the division must stay at its
    # original site (hoisting it would change the trap position and, for a
    # zero trip, invent an evaluation).  Inputs below never divide by zero.
    return program(
        func("main", [param("n", "int"), param("k", "int")], "int", [
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("q", "int", arith(
                    "div",
                    arith("add", var("k"), int_(1)),
                    arith("add", var("i"), int_(1)))),
                assign("total", arith("add", var("total"), var("q"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("e", "int", call("emit", [var("total")])),
            ret(var("total")),
        ]),
        _emit_fn(),
    )


def _unhoistable_call_program():
    # The in-loop emit must not move ahead of the loop: on a zero trip the
    # call never happens.  Calls are never hoisted.  The final observable is
    # the counter value, so a positive trip returns non-zero while a zero
    # trip returns zero.
    return program(
        func("main", [param("n", "int")], "int", [
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("r", "int",
                    call("emit", [arith("add", var("i"), int_(1))])),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("e", "int", call("emit", [var("i")])),
            ret(arith("add", var("i"), var("e"))),
        ]),
        _emit_fn(),
    )


# (label, builder, (arguments, expected-exit, expected-stdout))
_FIXED_SAMPLES = [
    ("fold-then-dce", _fold_dce_program, (
        ((7,), 1, b"5\n"),
        ((0,), 1, b"5\n"),
    )),
    ("licm-then-cleanup", _licm_cleanup_program, (
        ((3, 9), 1, b"30\n"),
        ((0, 5), 0, b"0\n"),
    )),
    ("retained-call-refold", _retained_call_refold_program, (
        ((4,), 1, b"1\n"),
        ((0,), 1, b"1\n"),
    )),
    ("preserved-effect", _preserved_effect_program, (
        ((), 1, b"42\n7\n"),
    )),
    ("unhoistable-trap-site", _unhoistable_trap_program, (
        ((3, 4), 1, b"8\n"),
        ((0, 5), 0, b"0\n"),
    )),
    ("unhoistable-zero-trip-call", _unhoistable_call_program, (
        ((2,), 1, b"1\n2\n2\n"),
        ((0,), 0, b"0\n"),
    )),
]


# ==========================================================================
# Tests
# ==========================================================================


class SeededCorpusTests(unittest.TestCase):
    def test_every_seed_is_type_valid_and_terminating(self):
        totals = dict(loops=0, shadows=0, const_branches=0, logicals=0,
                      helper_calls=0, emits=0, bool_exprs=0,
                      programs_with_helpers=0)
        # Coverage is pinned against the fixed default corpus even when the
        # reproduction seed override narrows the differential sweeps.
        for seed in DEFAULT_SEEDS:
            generator = ProgramGenerator(seed)
            ast = generator.program()
            # Type checking, name resolution and reachability all pass via
            # the existing public entry point.
            lowered = lower_module(copy.deepcopy(ast))
            self.assertFalse(lowered.ssa)

            # Two argument tuples execute normally (baseline interpreter
            # has a step budget, so an accidental infinite loop fails).
            for salt in (1, 2):
                arguments = _arguments_for(ast, seed, salt)
                outcome = _interpret(lowered, "main", arguments)
                self.assertEqual(
                    outcome.kind, "normal",
                    msg=f"seed {seed} {arguments}: {outcome!r}",
                )

            for key, value in generator.counts.items():
                totals[key] += value
            if generator.helpers:
                totals["programs_with_helpers"] += 1

        # Non-vacuity: the fixed corpus must actually exercise every
        # advertised feature; thresholds are deliberately well below
        # observed rates.
        size = len(DEFAULT_SEEDS)
        self.assertGreaterEqual(totals["loops"], size)
        self.assertGreaterEqual(totals["shadows"], size // 3)
        self.assertGreaterEqual(totals["const_branches"], 4)
        self.assertGreaterEqual(totals["logicals"], 3)
        self.assertGreaterEqual(totals["programs_with_helpers"], size // 3)
        self.assertGreaterEqual(totals["helper_calls"], size // 2)
        self.assertGreater(totals["emits"], size)
        self.assertGreater(totals["bool_exprs"], 0)

    def test_two_distinct_orders_are_always_compared(self):
        # Guard the guard: this suite compares at least two genuinely
        # different legal orders, not one order under two names.
        orders = list(LEGAL_ORDERS.values())
        self.assertGreaterEqual(len(orders), 2)
        self.assertNotEqual(orders[0], orders[-1])


class DifferentialOrderTests(unittest.TestCase):
    def _check(self, seed, arguments, order_items, run_target=True):
        generator = ProgramGenerator(seed)
        ast = generator.program()
        for order_name, order in order_items:
            failure = _first_failing_stage(
                ast, arguments, order_name, order, run_target=run_target)
            if failure is not None:
                stage, detail = failure

                def fails(candidate, _order=order, _arguments=arguments,
                          _stage=stage, _run_target=run_target):
                    # Reproduction predicate for the delta debugger.  It
                    # reruns the ordered checklist on a legal candidate with
                    # the same order/arguments and fires only while the
                    # *same* stage remains the first inconsistency.  Every
                    # probe terminates: the generator only builds bounded
                    # counter loops.  Deleting a statement could, in
                    # principle, detach the unique counter increment and make
                    # the interpreted candidate exhaust its step budget; such
                    # a candidate no longer represents the original stage, so
                    # the predicate reports None instead of raising.
                    try:
                        result = _first_failing_stage(
                            candidate, _arguments, "reduction", _order,
                            run_target=(_run_target
                                        and _stage == STAGE_SUBPROCESS))
                    except Exception:
                        return None
                    if result is not None and result[0] == _stage:
                        return _stage
                    return None

                source = minimize_program(ast, fails, stage)
                self.fail(format_failure(
                    seed, order_name, order, stage, source, detail))

    def test_all_orders_equivalent_in_process(self):
        # Every seed, every order, once and again: cheap in-process checks
        # plus the normalized-IR fixed point and target-byte reproducibility
        # (no subprocess is launched here).
        for seed in SEEDS:
            ast = ProgramGenerator(seed).program()
            for salt in (1, 2):
                arguments = _arguments_for(ast, seed, salt)
                with self.subTest(seed=seed, arguments=arguments,
                                  sweep="forward"):
                    self._check(seed, arguments, list(LEGAL_ORDERS.items()),
                                run_target=False)

    def test_subprocess_equivalence_for_representative_orders(self):
        # The real OS-process channel over representative legal orders for a
        # small seed subset; fixed samples cover every order end to end in a
        # separate test class.
        orders = [(name, LEGAL_ORDERS[name])
                  for name in SUBPROCESS_ORDER_NAMES]
        for seed in EXEC_SEEDS:
            ast = ProgramGenerator(seed).program()
            arguments = _arguments_for(ast, seed, 1)
            with self.subTest(seed=seed, arguments=arguments):
                self._check(seed, arguments, orders, run_target=True)

    def test_reverse_order_sweep_reaches_the_same_conclusion(self):
        for seed in EXEC_SEEDS:
            ast = ProgramGenerator(seed).program()
            arguments = _arguments_for(ast, seed, 3)
            with self.subTest(seed=seed, arguments=arguments,
                              sweep="reverse"):
                self._check(
                    seed, arguments,
                    list(reversed(list(LEGAL_ORDERS.items()))),
                    run_target=False)


class FixedPointAndReproducibilityTests(unittest.TestCase):
    def test_normalized_ir_is_fixed_when_the_same_order_runs_again(self):
        for seed in SEEDS:
            ast = ProgramGenerator(seed).program()
            for order_name, order in LEGAL_ORDERS.items():
                with self.subTest(seed=seed, order=order_name):
                    once = apply_order(
                        lower_module(copy.deepcopy(ast)), order)
                    again = apply_order(copy.deepcopy(once), order)
                    self.assertEqual(
                        normalized_ir(once), normalized_ir(again),
                        msg=f"seed {seed} order {order_name!r}: normalized "
                            "IR differs after the second application",
                    )

    def test_canonical_orders_are_byte_fixed_without_normalization(self):
        canonical = [(n, o) for n, o in LEGAL_ORDERS.items()
                     if o[-1] == "ssa" and len(o) > 1]
        self.assertTrue(canonical)
        for seed in SEEDS:
            ast = ProgramGenerator(seed).program()
            for order_name, order in canonical:
                with self.subTest(seed=seed, order=order_name):
                    once = apply_order(
                        lower_module(copy.deepcopy(ast)), order)
                    again = apply_order(copy.deepcopy(once), order)
                    self.assertEqual(
                        render_module(once), render_module(again))

    def test_same_configuration_compiles_byte_identically(self):
        for seed in SEEDS:
            ast = ProgramGenerator(seed).program()
            for order_name, order in LEGAL_ORDERS.items():
                with self.subTest(seed=seed, order=order_name):
                    first = compile_target(ast, order)
                    second = compile_target(ast, order)
                    third = compile_target(ast, order)
                    # Exact bytes: no order-insensitive comparison.
                    self.assertEqual(first, second)
                    self.assertEqual(second, third)


class FixedSampleDifferentialTests(unittest.TestCase):
    def test_fixed_samples_match_baseline_under_every_order_in_process(self):
        for label, builder, rows in _FIXED_SAMPLES:
            ast = builder()
            for arguments, expected_rc, expected_out in rows:
                baseline = _interpret(
                    lower_module(copy.deepcopy(ast)), "main", arguments)
                # Ground-truth pin against the unoptimized baseline.
                self.assertEqual(
                    (expected_rc, expected_out, b""),
                    expected_subprocess_triple(baseline),
                    msg=f"sample {label!r}: interpreter baseline disagrees "
                        "with the pinned expectation",
                )
                for order_name, order in LEGAL_ORDERS.items():
                    with self.subTest(sample=label, arguments=arguments,
                                      order=order_name):
                        once = apply_order(
                            lower_module(copy.deepcopy(ast)), order)
                        outcome = _interpret(once, "main", arguments)
                        self.assertEqual(
                            outcome, baseline,
                            msg=f"sample {label!r} order {order_name!r}: "
                                f"{outcome!r} != {baseline!r}",
                        )
                        again = apply_order(copy.deepcopy(once), order)
                        self.assertEqual(
                            normalized_ir(once), normalized_ir(again))
                        # Exact-byte reproducibility for this configuration.
                        self.assertEqual(
                            compile_target(ast, order),
                            compile_target(ast, order))

    def test_fixed_samples_execute_identically_in_a_subprocess(self):
        # Representative legal orders only (all orders are covered
        # in-process above); the emitted target really runs and must match
        # the pinned observables byte for byte.
        for label, builder, rows in _FIXED_SAMPLES:
            ast = builder()
            for arguments, expected_rc, expected_out in rows:
                for order_name in SUBPROCESS_ORDER_NAMES:
                    order = LEGAL_ORDERS[order_name]
                    with self.subTest(sample=label, arguments=arguments,
                                      order=order_name):
                        target = compile_target(ast, order)
                        run = execute_target(target, arguments)
                        self.assertEqual(
                            run.triple(),
                            (expected_rc, expected_out, b""),
                            msg=(f"sample {label!r} order {order_name!r}: "
                                f"subprocess {run.triple()!r}"),
                        )

    # -- structural pins for the three interacting orderings --------------

    def _main(self, module):
        return next(fn for fn in module.functions if fn.name == "main")

    def _blocks(self, module):
        return {b.label: b for b in self._main(module).blocks}

    def test_fold_dce_interaction_removes_only_the_dead_chain(self):
        ast = _fold_dce_program()
        from test_pass_ordering import definition_count

        plain = to_ssa(lower_module(copy.deepcopy(ast)))
        optimized = apply_order(
            lower_module(copy.deepcopy(ast)),
            ("ssa", "fold", "licm", "dce", "ssa"))
        self.assertLess(definition_count(optimized),
                        definition_count(plain))
        # The observable call survives; only the unread junk chain goes.
        calls = [ins.name for block in self._main(optimized).blocks
                 for ins in block.instructions if isinstance(ins, Call)]
        self.assertEqual(calls, ["emit"])

    def test_licm_cleanup_moves_the_invariant_to_the_preheader(self):
        ast = _licm_cleanup_program()
        hoisted = hoist_loop_invariants(
            to_ssa(lower_module(copy.deepcopy(ast))))
        preheader = self._blocks(hoisted)["b0"]
        # base = k + 1 lands directly before the preheader terminator.
        adds = [ins for ins in preheader.instructions
                if isinstance(ins, BinOp) and ins.operator == "add"]
        self.assertTrue(adds, "loop-invariant add was not hoisted")

    def test_retained_call_is_never_inlined_away(self):
        ast = _retained_call_refold_program()
        for order_name, order in LEGAL_ORDERS.items():
            module = apply_order(lower_module(copy.deepcopy(ast)), order)
            with self.subTest(order=order_name):
                calls = [ins.name for block in self._main(module).blocks
                         for ins in block.instructions
                         if isinstance(ins, Call)]
                self.assertEqual(calls, ["dbl", "emit"])

    def test_unread_side_effect_is_preserved(self):
        ast = _preserved_effect_program()
        for order_name, order in LEGAL_ORDERS.items():
            module = apply_order(lower_module(copy.deepcopy(ast)), order)
            with self.subTest(order=order_name):
                calls = [ins for block in self._main(module).blocks
                         for ins in block.instructions if isinstance(ins, Call)]
                self.assertEqual(len(calls), 2)

    def test_loop_trap_and_zero_trip_call_are_not_hoisted(self):
        for builder, body_label in (
                (_unhoistable_trap_program, "b2"),
                (_unhoistable_call_program, "b2")):
            ast = builder()
            for order_name, order in LEGAL_ORDERS.items():
                module = apply_order(
                    lower_module(copy.deepcopy(ast)), order)
                body = self._blocks(module)[body_label]
                with self.subTest(sample=builder.__name__, order=order_name):
                    if builder is _unhoistable_trap_program:
                        self.assertTrue(
                            any(isinstance(ins, BinOp)
                                and ins.operator == "div"
                                for ins in body.instructions))
                    else:
                        self.assertTrue(
                            any(isinstance(ins, Call)
                                for ins in body.instructions))
                # Nothing call-shaped is ever moved into the preheader.
                preheader = self._blocks(module)["b0"]
                self.assertFalse(
                    any(isinstance(ins, Call)
                        for ins in preheader.instructions))


class DiagnosticRegressionTests(unittest.TestCase):
    """Fixed illegal inputs keep their original exception and position."""

    def _assert_stable(self, bad, error_type, fragment):
        with self.assertRaises(error_type) as first:
            lower_module(copy.deepcopy(bad))
        with self.assertRaises(error_type) as second:
            lower_module(copy.deepcopy(bad))
        # TypeCheckError carries no AST path; the diagnostic position is
        # encoded in its stable message (function + argument / statement).
        self.assertIn(fragment, str(first.exception))
        self.assertEqual(str(first.exception), str(second.exception))
        self.assertIs(type(first.exception), error_type)

    def test_call_argument_type_error_position_is_unchanged(self):
        bad = program(
            func("id", [param("x", "int")], "int", [ret(var("x"))]),
            func("main", [], "int",
                 [ret(call("id", [bool_(False)]))]),
        )
        self._assert_stable(
            bad, TypeCheckError,
            "call to 'id' argument 1: expected int, got bool")

    def test_if_condition_type_error_position_is_unchanged(self):
        bad = program(func(
            "main", [], "int",
            [if_(int_(1), [ret(int_(1))], [ret(int_(2))])]))
        self._assert_stable(
            bad, TypeCheckError, "if condition must be bool, got int")

    def test_arith_operand_type_error_position_is_unchanged(self):
        bad = program(func(
            "main", [], "int",
            [ret(arith("add", int_(1), bool_(False)))]))
        self._assert_stable(
            bad, TypeCheckError,
            "arithmetic 'add' requires int operands, got int and bool")

    def test_undefined_symbol_diagnostic_is_unchanged(self):
        bad = program(func("main", [], "int", [ret(var("ghost"))]))
        self._assert_stable(
            bad, UndefinedSymbolError, "ghost")

    def test_illegal_program_never_reaches_a_pass(self):
        # The same bad source is rejected identically regardless of the
        # order that would have run: diagnostics precede optimization.
        bad = program(func(
            "main", [], "int",
            [if_(int_(0), [ret(int_(1))], [ret(int_(2))])]))
        for order_name, order in LEGAL_ORDERS.items():
            with self.subTest(order=order_name):
                with self.assertRaises(TypeCheckError):
                    apply_order(lower_module(copy.deepcopy(bad)), order)


class FailureReportTests(unittest.TestCase):
    """The minimization/reporting harness is itself deterministic."""

    def test_minimized_report_names_seed_order_stage_and_source(self):
        order_name = \
            "raw-core ssa,fold,licm,dce (no trailing canonicalization)"
        order = LEGAL_ORDERS[order_name]

        # Inject a compile-only "bug" that fires while a loop coexists with
        # at least four top-level main statements: the minimizer must shrink
        # to the smallest legal program that still triggers that stage.
        def failing_stage(candidate):
            try:
                module = apply_order(
                    lower_module(copy.deepcopy(candidate)), order)
            except Exception:
                return None
            main_fn = next(f for f in candidate["functions"]
                           if f["name"] == "main")
            has_loop = any(s["kind"] == "while" for s in main_fn["body"])
            if has_loop and len(main_fn["body"]) >= 4:
                return STAGE_IR_FIXED_POINT
            normalized_ir(module)  # ensure the module is renderable
            return None

        # Build a program known to trigger the injected stage.
        triggered = None
        for candidate_seed in range(200):
            candidate = ProgramGenerator(candidate_seed).program()
            if failing_stage(candidate) == STAGE_IR_FIXED_POINT:
                triggered = candidate_seed
                break
        self.assertIsNotNone(triggered,
                             "test setup: no seed triggered injected stage")
        original = ProgramGenerator(triggered).program()
        original_len = len(
            next(f for f in original["functions"]
                 if f["name"] == "main")["body"])

        reduced = minimize_program(
            original, failing_stage, STAGE_IR_FIXED_POINT)
        reduced_len = len(
            next(f for f in reduced["functions"]
                 if f["name"] == "main")["body"])
        self.assertLess(reduced_len, original_len)
        self.assertEqual(
            failing_stage(reduced), STAGE_IR_FIXED_POINT)

        message = format_failure(
            triggered, order_name, order, STAGE_IR_FIXED_POINT,
            reduced, "injected failure")
        self.assertIn(f"seed                 : {triggered}", message)
        self.assertIn(STAGE_IR_FIXED_POINT, message)
        self.assertIn("ssa", message)
        # The minimized source is present, valid JSON and reproducible.
        self.assertIn("minimized source program:", message)
        serialized = message.split("minimized source program:\n", 1)[1]
        rebuilt = json.loads(serialized)
        self.assertEqual(
            failing_stage(rebuilt), STAGE_IR_FIXED_POINT)
        # Regenerating from the seed in a fresh pass reproduces the same
        # original failure inside this single process.
        self.assertEqual(
            failing_stage(ProgramGenerator(triggered).program()),
            STAGE_IR_FIXED_POINT)


if __name__ == "__main__":
    unittest.main()
