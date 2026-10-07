"""Differential and metamorphic tests for the optimisation pipeline.

One and the same *legal* source program must keep its observable semantics
under every legal ordering of the existing public optimisation passes,
under repetition of the whole sequence, and it must compile to
byte-identical target code for a fixed source/configuration.  Nothing in
the compiler package is changed: every artifact is obtained through the
existing public entry points

    ``lower_module`` -> ``optimize_module`` (a legal pass order) ->
    ``render_module``

and every generated program is accepted by the existing type rules, uses
only the documented AST subset, and is guaranteed to terminate on the
argument domain it is generated for.  Illegal inputs are deliberately kept
out of the semantic comparison; the pre-existing diagnostics have their
own fixed regression cases below.

Three independent oracles agree
--------------------------------

1. the *reference semantics* -- a small Python implementation written next
   to each program template, pinning the return value and the ordered
   ``emit`` output from source semantics alone;
2. the repository's existing in-process IR interpreter
   (``test_pass_ordering._interpret``), run on the **unoptimized**
   ``lower_module`` result (the baseline) and again on every optimized
   module, including div/mod-by-zero fault outcomes;
3. the repository's existing execution harness
   (``test_pass_order_execution.execute_target``), which renders the SSA
   target and runs it as a real operating-system subprocess, comparing the
   raw exit status / stdout / stderr bytes.

The normalized terminal IR is ``render_module(to_ssa(module))``: the public
``ssa`` stage on an SSA module is the deterministic renumbering
canonicalization, so comparison is strict, order-sensitive text equality --
never a comparison that sorts or ignores ordering, which could mask
nondeterminism.

For every legal program and every order the suite checks that

* the order executed once matches the unoptimized baseline through both
  execution entries;
* the order executed a second time on the optimized result keeps the same
  interpreted behavior and the same *normalized* IR as the first
  execution (raw text may legitimately differ when the order stops short
  of the trailing canonicalization; the normalized form may not);
* independently recompiling the same source/order three times yields
  byte-identical target code and stable serialized text.

The orders in the main table are all self-fixed-points.  One extra order
-- LICM scheduled *before* any fold -- is a deliberate, documented
exception to the one-shot idempotence claim: the pass's fault-site fence
keeps an un-folded loop operand in place on the first application, so a
strictly larger (but still semantics-preserving) set moves only once
folding has exposed the constant.  A dedicated metamorphic test therefore
requires identical behavior on the first, second and third applications,
requires at least one program to exhibit the extra motion, and requires
the normalized IR to settle from the second application onward.

Corpus
------

Fixed hand-written programs concentrate the pass interactions the task
calls out -- constant propagation followed by DCE, loop-invariant motion
followed by cleanup, and an inline-shaped call followed by further folding
-- and, additionally, a side-effecting expression that no legal order may
remove, a loop computation that is not safe to hoist (loop-carried and a
guarded ``div``), and a division trap whose site must not move.  A seeded
generator then produces terminating, well-typed variants covering int and
bool expressions, locals and shadowing, conditional branches (including a
folded-to-constant "invalid" arm that still contains calls and arithmetic
at compile time), bounded loops with loop invariants, and function calls.

Reproduction
------------

Every generated/enumerated choice accepts an explicit integer seed
(``DIFF_TEST_SEED`` overrides the pinned default).  A failure report
contains the seed, the minimized source program (an AST delta search under
a fixed attempt budget, using only in-process checks), the pass order and
the *first* disagreement stage, so the same failure replays inside one
test process.
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
    Const,
    DuplicateSymbolError,
    InvalidAstError,
    MissingReturnError,
    TypeCheckError,
    UndefinedSymbolError,
    lower_module,
    optimize_module,
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
    DEFAULT_ORDER,
    RunResult,
    compile_target,
    execute_target,
)


# Pinned seed; CI may replay another seed via DIFF_TEST_SEED.
SEED = int(os.environ.get("DIFF_TEST_SEED", "20241007"))

# How many argument tuples the seeded generator draws per program.
ARG_DRAWS = 2


# ==========================================================================
# Legal pass orders (public names only; preconditions satisfied)
# ==========================================================================
#
# Every order begins with "ssa" (fold/licm/dce require it on a non-SSA
# input), names may repeat, and render_module is never part of an order.
# Orders ending in "ssa" are canonical fixed points; the RAW_CORE order
# deliberately stops before the trailing renumbering, so its emitted
# bytes may differ while its behavior and normalized IR must not.
ORDERS = [
    ("default ssa,fold,licm,dce,ssa",
     ("ssa", "fold", "licm", "dce", "ssa")),
    ("repeat-fold ssa,fold,fold,licm,dce,ssa",
     ("ssa", "fold", "fold", "licm", "dce", "ssa")),
    ("dce-before-fold ssa,dce,fold,licm,dce,ssa",
     ("ssa", "dce", "fold", "licm", "dce", "ssa")),
    ("interleaved ssa,fold,dce,licm,fold,dce,ssa",
     ("ssa", "fold", "dce", "licm", "fold", "dce", "ssa")),
    ("fold-after-cleanup ssa,fold,licm,dce,fold,dce,ssa",
     ("ssa", "fold", "licm", "dce", "fold", "dce", "ssa")),
    ("raw-core ssa,fold,licm,dce",
     ("ssa", "fold", "licm", "dce")),
    ("dce-licm-adjacent ssa,fold,dce,dce,licm,ssa",
     ("ssa", "fold", "dce", "dce", "licm", "ssa")),
    ("short-canonical ssa,fold,dce,ssa",
     ("ssa", "fold", "dce", "ssa")),
]
RAW_CORE_INDEX = 5

# A deliberately non-idempotent order: LICM before any fold.  Its fault
# fence keeps an un-folded ``add``/``sub``/``mul`` operand inside a loop;
# once folding has exposed the constant, a second application can hoist
# more.  The order therefore *converges* (it reaches a fixed point on the
# second application) rather than being a fixed point after one; a
# dedicated metamorphic test below pins both the semantic preservation at
# every application and that convergence.
LICM_FIRST_ORDER = (
    "licm-first ssa,licm,fold,dce,fold,dce,ssa",
    ("ssa", "licm", "fold", "dce", "fold", "dce", "ssa"),
)


# ==========================================================================
# Small shared AST fragments
# ==========================================================================


def emit_function():
    """The reserved stdout channel: emit(int) -> int (identity)."""
    return func("emit", [param("v", "int")], "int", [ret(var("v"))])


def trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


# ==========================================================================
# Hand-written fixed programs
# ==========================================================================
#
# Each builder returns ``(ast, [(arguments, expected_return,
# expected_emits)])``; expectations are pinned from source semantics.  All
# programs terminate on every listed argument tuple.


def fold_dce_program():
    # Constant propagation first, then DCE:
    #   k = 1 + 2 folds to 3; junk = (k * 4) - 2 is a pure chain with no
    #   observable use and disappears only after folding exposes it; the
    #   unused emit result must still be emitted (a call is a root), and
    #   the folded offset 2 * 3 feeding the live emit folds with it.
    ast = program(
        func("main", [param("n", "int")], "int", [
            let("k", "int", arith("add", int_(1), int_(2))),
            let("junk", "int",
                arith("sub", arith("mul", var("k"), int_(4)), int_(2))),
            let("ignored", "int", call("emit", [int_(77)])),
            let("r", "int",
                arith("add", var("k"), arith("mul", int_(2), int_(3)))),
            let("e", "int", call("emit", [var("r")])),
            ret(arith("add", arith("add", var("r"), var("n")), var("e"))),
        ]),
        emit_function(),
    )
    # r = 3 + 6 = 9; emits 77 then 9; return 9 + n + 9.
    rows = [((4,), 22, [77, 9]), ((-9,), 9, [77, 9])]
    return ast, rows


def licm_cleanup_program():
    # Loop-invariant motion followed by cleanup:
    #   (10 + 5) folds and is invariant; its hoisted Const 15 lands in the
    #   loop preheader.  pure = (10 + 5) * k is invariant too and is moved
    #   out, while junk = (2 + 3) * pure is dead and is reclaimed by the
    #   cleanup DCE.  The loop-carried accumulation (i) stays in the body;
    #   emit(i) pins per-iteration output and is never hoisted.
    ast = program(
        func("main", [param("n", "int"), param("k", "int")], "int", [
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("ho", "int",
                    arith("add", arith("add", int_(10), int_(5)), var("i"))),
                let("pure", "int",
                    arith("mul", arith("add", int_(10), int_(5)), var("k"))),
                let("junk", "int",
                    arith("mul", arith("add", int_(2), int_(3)), var("pure"))),
                let("pe", "int", call("emit", [var("i")])),
                assign("total",
                       arith("add", arith("add", var("total"), var("ho")),
                             var("pure"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("e", "int", call("emit", [var("total")])),
            ret(arith("add", var("total"), var("e"))),
        ]),
        emit_function(),
    )
    # iteration i contributes ho = 15 + i and pure = 15*k.
    rows = []
    for n, k in ((2, 1), (0, 4), (3, -1)):
        emits = list(range(n))
        total = sum((15 + i) + 15 * k for i in range(n))
        emits.append(total)
        rows.append(((n, k), 2 * total, emits))
    return ast, rows


def inline_then_fold_program():
    # Inline-shaped callee followed by further constant folding:
    #   no implemented pass inlines "dbl", so the call and its stream
    #   position must survive every order; (1 + 1) inside dbl folds after
    #   the (absent) inlining point, and the caller's post-call constants
    #   fold around its unknown result.  junk1/junk2 are post-call dead
    #   arithmetic removed only by the cleanup DCE.
    ast = program(
        func("main", [param("n", "int")], "int", [
            let("v", "int", call("dbl", [arith("add", var("n"), int_(1))])),
            let("e0", "int", call("emit", [var("v")])),
            let("junk1", "int", arith("mul", var("v"), int_(9))),
            let("junk2", "int", arith("add", var("junk1"), int_(5))),
            let("w", "int",
                arith("add", var("v"), arith("mul", int_(2), int_(3)))),
            let("e1", "int", call("emit", [int_(1)])),
            ret(var("w")),
        ]),
        func("dbl", [param("x", "int")], "int", [
            let("two", "int", arith("add", int_(1), int_(1))),
            let("unused", "int", arith("mul", var("two"), int_(9))),
            ret(arith("mul", var("x"), var("two"))),
        ]),
        emit_function(),
    )
    # v = 2*(n+1); emits v then 1; w = v + 6.
    rows = [((4,), 16, [10, 1]), ((0,), 8, [2, 1]), ((-2,), 4, [-2, 1])]
    return ast, rows


def nonhoistable_loop_program():
    # A loop computation that is NOT safe to hoist:
    #   safe = (2 + 3) * i folds its constant part, but the multiply is
    #   loop-carried through i and stays in the body; r = 100 div (4 - i)
    #   is both loop-dependent and a potentially trapping div, so LICM
    #   must leave it at its original site (the chosen domain makes the
    #   divisor 3, 2, 1 -- never zero).  q = 20 div (n + 4) is outside
    #   the loop and stays a div BinOp as well.
    ast = program(
        func("main", [param("n", "int")], "int", [
            let("total", "int", int_(0)),
            let("i", "int", int_(1)),
            while_(logical("and",
                           compare("le", var("i"), var("n")),
                           compare("le", var("i"), int_(3))), [
                let("safe", "int",
                    arith("mul", arith("add", int_(2), int_(3)), var("i"))),
                let("r", "int",
                    arith("div", int_(100),
                          arith("sub", int_(4), var("i")))),
                let("pe", "int", call("emit", [var("safe")])),
                assign("total", arith(
                    "add", arith("add", var("total"), var("safe")), var("r"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("q", "int", arith("div", int_(20),
                                  arith("add", var("n"), int_(4)))),
            let("eq", "int", call("emit", [var("q")])),
            ret(arith("add", var("total"), var("eq"))),
        ]),
        emit_function(),
    )
    rows = []
    for n in (-1, 0, 2, 3):
        iters = max(min(n, 3), 0)
        emits = [5 * i for i in range(1, min(n, 3) + 1)]
        total = sum(5 * i + trunc_div(100, 4 - i)
                    for i in range(1, min(n, 3) + 1))
        q = trunc_div(20, n + 4)
        emits.append(q)
        rows.append(((n,), total + q, emits))
    return ast, rows


def guarded_loop_trap_program():
    # The div is reached only after an observable emit and only on the
    # iteration where 2 - i == 0 (i == 2); folding must not advance or
    # swallow the trap and LICM must not move it.  n == 1 runs one
    # iteration (100 div 2 = 50); n == 3 faults on iteration 2 after the
    # prefix emits 0, 1, 2.
    ast = program(
        func("main", [param("n", "int")], "int", [
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("pe", "int", call("emit", [var("i")])),
                let("q", "int",
                    arith("div", int_(100),
                          arith("sub", int_(2), var("i")))),
                assign("total", arith("add", var("total"), var("q"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            ret(var("total")),
        ]),
        emit_function(),
    )
    return ast, [((1,), 50, [0]), ((2,), 150, [0, 1])]


FIXED_PROGRAMS = [
    ("fold-then-dce", fold_dce_program),
    ("licm-then-cleanup", licm_cleanup_program),
    ("inline-shaped-then-fold", inline_then_fold_program),
    ("nonhoistable-loop-computation", nonhoistable_loop_program),
]


# ==========================================================================
# Seeded program generator
# ==========================================================================
#
# Every template is generated together with a Python reference function so
# the expected observable result is derived independently of the compiler.
# All loops are bounded either by an explicit cap or by an argument domain
# that is itself bounded; every generated program calls emit, so stdout is
# never empty by coincidence.


def _template_expressions_shadows(variant: int, a: int, b: int, c: bool):
    # Int and bool expressions, an inner same-name shadow, a merge phi,
    # short-circuit logic and a branch folded to a constant whose dead arm
    # still contains arithmetic plus a (never executed) call.
    junk_const = (7, 2, 3, 4) if variant == 0 else (8, 1, 2, 2)
    cond = (logical("and", var("c"), compare("gt", var("a"), int_(-5)))
            if variant == 0 else
            logical("or", var("c"), compare("lt", var("a"), int_(-10))))
    ast = program(
        func("main",
             [param("a", "int"), param("b", "int"), param("c", "bool")],
             "int", [
            let("k", "int",
                arith("mul", arith("add", var("a"), var("b")), int_(2))),
            let("m", "int", arith("sub", var("k"), int_(3))),
            let("junk", "int", arith(
                "mul",
                arith("sub", int_(junk_const[0]), int_(junk_const[1])),
                arith("add", int_(junk_const[2]), int_(junk_const[3])))),
            let("e0", "int",
                call("emit", [arith("add", var("k"), var("m"))])),
            let("x", "int", arith("add", var("a"), int_(1))),
            block([
                let("x", "int", arith("add", var("b"), int_(2))),
                let("sx", "int", call("emit", [var("x")])),
            ]),
            let("ox", "int", call("emit", [var("x")])),
            let("v", "int", int_(0)),
            if_(cond,
                [assign("v", arith("add", var("k"), var("x")))],
                [assign("v", arith("sub", var("m"), var("x")))]),
            if_(compare("eq", arith("add", int_(2), int_(2)), int_(4)),
                [let("t", "int", call("emit", [int_(11)]))],
                [let("dd", "int", arith("mul", int_(9), int_(9))),
                 let("dc", "int", call("emit", [int_(999)]))]),
            let("ev", "int", call("emit", [var("v")])),
            ret(arith("add", arith("add", var("v"), var("ev")), var("ox"))),
        ]),
        emit_function(),
    )

    def reference():
        emits = []
        emit = emits.append
        k = (a + b) * 2
        m = k - 3
        emit(k + m)
        x = a + 1
        emit(b + 2)              # shadowed x inside the block
        emit(x)                  # outer x visible again
        takes_true = (c and a > -5) if variant == 0 else (c or a < -10)
        v = (k + x) if takes_true else (m - x)
        emit(11)                 # constant-folded branch takes the true arm
        emit(v)
        return v + v + x, emits

    return ast, reference()


def _template_bounded_loop(variant: int, n: int, k: int):
    # Terminating loop with a header phi carried over the back edge, a
    # branch inside the body, short-circuit loop condition, a dead
    # invariant chain, and a shadowing block after the exit.
    cap = 4 if variant == 0 else 3
    shadow_value = 33 if variant == 0 else 44
    junk_pair = ((2, 3), (4, 6))[variant]
    cond = (logical("and",
                    compare("lt", var("i"), var("n")),
                    compare("lt", var("i"), int_(cap)))
            if variant == 0 else
            compare("lt", var("i"), var("n")))
    ast = program(
        func("main", [param("n", "int"), param("k", "int")], "int", [
            let("base", "int", arith("add", var("k"), int_(1))),
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(cond, [
                let("pe", "int", call("emit", [var("i")])),
                if_(compare("gt", var("i"), int_(0)),
                    [assign("total", arith(
                        "add",
                        arith("add", var("total"), var("base")), var("i")))],
                    [let("z", "int", call("emit", [int_(7)])),
                     assign("total", arith("add", var("total"), var("z")))]),
                let("junk", "int", arith(
                    "mul",
                    arith("add", int_(junk_pair[0]), int_(junk_pair[1])),
                    var("k"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("et", "int", call("emit", [var("total")])),
            block([let("n", "int", call("emit", [int_(shadow_value)]))]),
            ret(arith("add", var("total"), var("et"))),
        ]),
        emit_function(),
    )

    def reference():
        emits = []
        total = 0
        limit = min(n, cap) if variant == 0 else n
        for i in range(max(limit, 0)):
            emits.append(i)
            if i > 0:
                total += (k + 1) + i
            else:
                emits.append(7)
                total += 7
        emits.append(total)
        emits.append(shadow_value)
        return 2 * total, emits

    return ast, reference()


def _template_calls(variant: int, a: int, b: int):
    # Function calls (including multi-function forward calls and calls in
    # both branches), constants folded around unknown call results, and
    # dead post-call arithmetic.  variant 1 makes clamp take its other arm.
    ast = program(
        func("main", [param("a", "int"), param("b", "int")], "int", [
            let("u", "int", call("dbl", [arith("add", var("a"), int_(1))])),
            let("eu", "int", call("emit", [var("u")])),
            let("v", "int", call("clamp", [var("b")])),
            let("ev", "int", call("emit", [var("v")])),
            let("junk1", "int", arith("mul", var("eu"), int_(9))),
            let("junk2", "int", arith("add", var("junk1"), int_(5))),
            let("w", "int",
                call("pick", [compare("lt", var("a"), var("b")),
                              var("u"), var("v")])),
            let("ew", "int", call("emit", [var("w")])),
            let("q", "int",
                arith("add",
                      arith("mul", int_(1), int_(2)), var("ew"))),
            let("dq", "int", arith("sub", var("q"), int_(3))),
            ret(arith("add", arith("add", var("u"), var("v")), var("ew"))),
        ]),
        func("dbl", [param("x", "int")], "int", [
            let("t", "int", arith("add", var("x"), var("x"))),
            ret(var("t")),
        ]),
        func("clamp", [param("x", "int")], "int", [
            if_(compare("lt", var("x"), int_(0)),
                [ret(int_(0))],
                [ret(var("x"))]),
        ]),
        func("pick",
             [param("c", "bool"), param("x", "int"), param("y", "int")],
             "int", [
            if_(var("c"), [ret(var("x"))], [ret(var("y"))]),
        ]),
        emit_function(),
    )

    def reference():
        emits = []
        u = (a + 1) + (a + 1)
        emits.append(u)
        v = 0 if b < 0 else b
        emits.append(v)
        w = u if a < b else v
        emits.append(w)
        return u + v + w, emits

    return ast, reference()


def _template_nested_loops(variant: int, n: int):
    # Nested terminating loops: a constant folded from (10 + 5) is
    # invariant to both loops and lands in the outermost preheader; the
    # j-carried and i-carried accumulations must stay in their loop
    # bodies.  The argument domain bounds both loops.
    pair = (10, 5) if variant == 0 else (4, 11)
    ast = program(
        func("main", [param("n", "int")], "int", [
            let("outer", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("inner", "int", int_(0)),
                let("j", "int", int_(0)),
                while_(compare("lt", var("j"), var("n")), [
                    let("c", "int",
                        arith("add",
                              arith("add", int_(pair[0]), int_(pair[1])),
                              int_(1))),
                    assign("inner", arith(
                        "add",
                        arith("add", var("inner"),
                              arith("add", var("j"), int_(1))),
                        arith("add",
                              arith("add", int_(pair[0]), int_(pair[1])),
                              int_(1)))),
                    let("pe", "int",
                        call("emit", [arith("add",
                                            arith("mul", var("i"), int_(10)),
                                            var("j"))])),
                    assign("j", arith("add", var("j"), int_(1))),
                ]),
                assign("outer", arith("add", var("outer"), var("inner"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            ret(var("outer")),
        ]),
        emit_function(),
    )

    def reference():
        emits = []
        outer = 0
        for i in range(max(n, 0)):
            inner = 0
            for j in range(max(n, 0)):
                inner += (j + 1) + (pair[0] + pair[1] + 1)
                emits.append(i * 10 + j)
            outer += inner
        return outer, emits

    return ast, reference()


def _template_loop_guarded_div(variant: int, n: int):
    # Bounded loop with a guarded, loop-dependent div (never zero on the
    # generated domain), a folded loop-invariant constant and an
    # after-loop div: none of the divs may be folded away or hoisted.
    const_pair = (2, 3) if variant == 0 else (1, 4)
    ast = program(
        func("main", [param("n", "int")], "int", [
            let("total", "int", int_(0)),
            let("i", "int", int_(1)),
            while_(logical("and",
                           compare("le", var("i"), var("n")),
                           compare("le", var("i"), int_(3))), [
                let("safe", "int", arith(
                    "mul",
                    arith("add", int_(const_pair[0]), int_(const_pair[1])),
                    var("i"))),
                let("r", "int",
                    arith("div", int_(100),
                          arith("sub", int_(4), var("i")))),
                let("pe", "int", call("emit", [var("safe")])),
                assign("total", arith(
                    "add", arith("add", var("total"), var("safe")), var("r"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            let("q", "int", arith("div", int_(20),
                                  arith("add", var("n"), int_(4)))),
            let("eq", "int", call("emit", [var("q")])),
            ret(arith("add", var("total"), var("eq"))),
        ]),
        emit_function(),
    )

    def reference():
        five = const_pair[0] + const_pair[1]
        emits = [five * i for i in range(1, min(n, 3) + 1)]
        total = sum(five * i + trunc_div(100, 4 - i)
                    for i in range(1, min(n, 3) + 1))
        q = trunc_div(20, n + 4)
        emits.append(q)
        return total + q, emits

    return ast, reference()


def _template_booleans(variant: int, c: bool, a: int):
    # Boolean literals, comparisons, both short-circuit operators, a
    # constant-folded branch (its dead arm keeps arithmetic plus a call
    # that never executes) and a dead pure integer chain.  The branch
    # condition depends on c (it is not folded), while the second branch
    # folds to a literal.  The domain keeps the short-circuit right side
    # observable for exactly one value of c.
    junk_pair = ((8, 1, 2, 2), (6, 2, 3, 1))[variant]
    cond = (logical("and", var("c"), compare("gt", var("a"), int_(0)))
            if variant == 0 else
            logical("or", var("c"), compare("lt", var("a"), int_(-10))))
    ast = program(
        func("main", [param("c", "bool"), param("a", "int")], "int", [
            let("e0", "int", call("emit", [int_(1)])),
            let("x", "int", int_(0)),
            if_(cond,
                [assign("x", arith("add", var("a"), int_(1)))],
                [assign("x", arith("sub", var("a"), int_(1)))]),
            let("ex", "int", call("emit", [var("x")])),
            if_(compare("lt", int_(3), int_(5)),
                [let("alive", "int", call("emit", [int_(2)]))],
                [let("dd", "int", arith("mul", int_(9), int_(9))),
                 let("dead", "int", call("emit", [int_(998)]))]),
            let("junk", "int", arith(
                "mul",
                arith("sub", int_(junk_pair[0]), int_(junk_pair[1])),
                arith("add", int_(junk_pair[2]), int_(junk_pair[3])))),
            ret(arith("add", var("ex"), var("e0"))),
        ]),
        emit_function(),
    )

    def reference():
        emits = [1]
        if variant == 0:
            takes_true = c and a > 0
        else:
            takes_true = c or a < -10
        x = a + 1 if takes_true else a - 1
        emits.append(x)
        emits.append(2)      # 3 < 5 folds to true; the false arm never runs
        return x + 1, emits

    return ast, reference()


# (template name, builder, [(variant, argument tuple domain)])
_TEMPLATES = [
    ("expressions-shadows-branches", _template_expressions_shadows, [
        (0, [(-3, -2, True), (-1, 3, False), (0, 0, True), (4, -2, False)]),
        (1, [(2, 3, False), (4, 0, True), (-3, 3, True)]),
    ]),
    ("bounded-loop-invariant", _template_bounded_loop, [
        (0, [(-2, -1), (0, 4), (2, 1), (3, -1), (5, 0)]),
        (1, [(0, 2), (2, -1), (3, 0)]),
    ]),
    ("function-calls", _template_calls, [
        (0, [(-2, -5), (0, 0), (3, 4)]),
        (1, [(3, 0), (-2, 4), (0, -5)]),
    ]),
    ("nested-loops", _template_nested_loops, [
        (0, [(0,), (1,), (2,)]),
        (1, [(0,), (2,)]),
    ]),
    ("loop-guarded-div", _template_loop_guarded_div, [
        (0, [(-1,), (0,), (2,), (3,)]),
        (1, [(-1,), (0,), (2,)]),
    ]),
    ("booleans-and-invalid-branch", _template_booleans, [
        (0, [(True, -1), (False, 0), (True, 3)]),
        (1, [(False, 3), (True, -1), (False, 0)]),
    ]),
]


def generate_corpus(seed: int):
    """Deterministically produce the seeded differential corpus.

    Returns ``[(label, ast, arguments, (expected_return, expected_emits),
    order_indices_for_twice_exec)]``.  Both variants of every template are
    included (feature coverage does not depend on the seed); the seed draws
    the argument tuples, rotates the order in which cases are visited and
    chooses which two orders get a second subprocess execution per case.
    """
    rng = random.Random(seed)
    cases = []
    for name, builder, variant_rows in _TEMPLATES:
        for variant, domain in variant_rows:
            # Sample up to ARG_DRAWS+1 distinct tuples without replacement so
            # small domains do not produce duplicate cases; fall back to a
            # seed-picked tuple when the domain is smaller than the quota.
            pool = list(domain)
            rng.shuffle(pool)
            chosen = pool[:min(ARG_DRAWS + 1, len(pool))]
            if len(chosen) <= ARG_DRAWS:
                chosen.append(domain[(variant + seed) % len(domain)])
                chosen = list(dict.fromkeys(chosen))
            for args in chosen:
                ast, expected = builder(variant, *args)
                # The deliberately un-canonicalized raw order always gets a
                # real process; about a third of cases additionally rerun a
                # seed-chosen order after repetition in a second process.
                exec_once = (RAW_CORE_INDEX,)
                exec_twice = ((rng.randrange(len(ORDERS)),)
                              if rng.random() < 1.0 / 3.0 else ())
                label = f"{name}#v{variant}{args}"
                cases.append((label, ast, args, expected,
                              exec_once, exec_twice))
    rng.shuffle(cases)
    return cases


# ==========================================================================
# Compilation through the public entry points
# ==========================================================================


def _lower(ast):
    return lower_module(copy.deepcopy(ast))


def _optimize(ast, order):
    return optimize_module(_lower(ast), order)


def _target(ast, order) -> bytes:
    # Same public path the existing execution suite uses, for byte-exact
    # comparisons against its artifacts.
    return compile_target(ast, order)


def _normalized(module) -> str:
    return render_module(to_ssa(module))


def _expected_triple(expected_return, expected_emits):
    # One line per executed emit; no emit call means zero bytes (not one
    # empty line), so the join-then-newline construction is avoided.
    stdout = "".join(f"{v}\n" for v in expected_emits).encode("utf-8")
    exit_status = 1 if expected_return != 0 else 0
    return exit_status, stdout, b""


# ==========================================================================
# Stage-wise differential check
# ==========================================================================
#
# Each check returns None on success or a (stage, detail) mismatch.  Stages
# run in a fixed order so a failure names the FIRST disagreement.


def _outcome_matches_reference(outcome, expected):
    expected_return, expected_emits = expected
    if outcome.kind != "normal":
        return ("unexpected fault",
                f"category={outcome.category!r} site={outcome.site!r}")
    # The observable output channel is emit; other calls are internal
    # computations whose results already feed the checked return value.
    calls = [(name, args) for name, args in outcome.output]
    if any(len(args) != 1 for name, args in calls if name == "emit"):
        return ("ill-formed emit trace", f"trace={calls!r}")
    emits = [args[0] for name, args in calls if name == "emit"]
    if emits != expected_emits:
        return ("stdout (ordered emit values)",
                f"expected={expected_emits!r} actual={emits!r}")
    if outcome.value != expected_return:
        return ("return value",
                f"expected={expected_return!r} actual={outcome.value!r}")
    return None


def _check_case(ast, args, expected, exec_once=frozenset(),
                exec_twice=frozenset(), subprocesses=True):
    """Run all differential stages for one case; return first mismatch.

    The in-process interpreter (the repository's existing evaluation
    entry) covers every order once *and* twice.  A real subprocess is the
    other existing entry and runs for the order indices in ``exec_once``
    (the default order always runs) and again after repetition for the
    indices in ``exec_twice``; it is deliberately sampled rather than run
    for every order so the seeded sweep stays fast, while the fixed
    regression samples every order.
    """
    expected_return, _emits = expected
    exec_once = set(exec_once) | {0}

    # Stage 0: the unoptimized lowering already agrees with the reference.
    baseline_module = _lower(ast)
    baseline = _interpret(baseline_module, "main", args)
    mismatch = _outcome_matches_reference(baseline, expected)
    if mismatch is not None:
        return ("baseline-reference", mismatch[0], mismatch[1])

    # Stage 1: the default-order target runs as a real process and meets
    # the pinned observables (raw bytes compared).
    if subprocesses:
        default_run = execute_target(_target(ast, DEFAULT_ORDER), args)
        wanted = RunResult(*_expected_triple(expected_return, _emits))
        if default_run.triple() != wanted.triple():
            return ("baseline-target", "default-order subprocess",
                    f"expected={wanted.triple()!r} "
                    f"actual={default_run.triple()!r}")

    for index, (order_name, order) in enumerate(ORDERS):
        once = _optimize(ast, order)

        # Stage 2a: interpreted once == unoptimized baseline.
        once_outcome = _interpret(once, "main", args)
        if once_outcome != baseline:
            return (order_name, "once-interpreter",
                    f"baseline={baseline!r} once={once_outcome!r}")

        if subprocesses and index in exec_once:
            # Stage 2b: executed target once == pinned observables.
            run_once = execute_target(_target(ast, order), args)
            wanted = RunResult(*_expected_triple(expected_return, _emits))
            if run_once.triple() != wanted.triple():
                return (order_name, "once-target",
                        f"expected={wanted.triple()!r} "
                        f"actual={run_once.triple()!r}")

        # Stage 3: the same sequence applied a second time keeps behavior
        # and reaches the same NORMALIZED IR (strict text equality; raw
        # text may differ for non-canonical endpoints, normalized may not).
        twice = optimize_module(once, order)
        twice_outcome = _interpret(twice, "main", args)
        if twice_outcome != baseline:
            return (order_name, "twice-interpreter",
                    f"baseline={baseline!r} twice={twice_outcome!r}")
        norm_once = _normalized(once)
        norm_twice = _normalized(twice)
        if norm_once != norm_twice:
            first = _first_text_difference(norm_once, norm_twice)
            return (order_name, "twice-normalized-ir",
                    f"normalized IR changed on repetition; {first}")

        if subprocesses and index in exec_twice:
            run_twice = execute_target(render_module(twice).encode("utf-8"),
                                       args)
            wanted = RunResult(*_expected_triple(expected_return, _emits))
            if run_twice.triple() != wanted.triple():
                return (order_name, "twice-target",
                        f"expected={wanted.triple()!r} "
                        f"actual={run_twice.triple()!r}")

        # Stage 4: same source/configuration recompiled independently is
        # byte-identical (three compilations, fresh lowering each time).
        recompiled = [_target(ast, order) for _ in range(3)]
        if recompiled[0] != recompiled[1] or recompiled[1] != recompiled[2]:
            diff = _first_text_difference(recompiled[0], recompiled[1]) \
                if recompiled[0] != recompiled[1] else \
                _first_text_difference(recompiled[1], recompiled[2])
            return (order_name, "recompile-byte-identity",
                    f"target bytes differ across recompilations; {diff}")

    return None


def _first_text_difference(a, b):
    if isinstance(a, str):
        a, b = a.encode("utf-8"), b.encode("utf-8")
    limit = min(len(a), len(b))
    offset = next((i for i in range(limit) if a[i] != b[i]), None)
    if offset is None:
        offset = limit if len(a) != len(b) else None
    if offset is None:
        return "texts differ without a differing byte (?!?)"
    line = a.count(b"\n", 0, offset) + 1
    line_start = a.rfind(b"\n", 0, offset) + 1
    line_end = a.find(b"\n", offset)
    return (f"first differing byte {offset} on line {line}: "
            f"{a[line_start:line_end]!r}")


# ==========================================================================
# AST delta minimization for failure reports
# ==========================================================================


def _ast_size(node) -> int:
    if isinstance(node, dict):
        return 1 + sum(_ast_size(v) for v in node.values())
    if isinstance(node, list):
        return sum(_ast_size(v) for v in node)
    return 0


def _main(ast):
    return next(f for f in ast["functions"] if f["name"] == "main")


def _minimize(ast, args, expected, order_name, stage, budget=80):
    """Shrink ``ast`` while it still reproduces the mismatch.

    Only deletions that lower successfully and keep the case failing are
    accepted; the search is purely in-process (no subprocess) and bounded.
    """
    def reproduces(candidate) -> bool:
        try:
            lower_module(copy.deepcopy(candidate))
        except Exception:
            return False
        result = _check_case(candidate, args, expected,
                             subprocesses=False)
        return result is not None and result[0] == order_name and \
            result[1] == stage

    best = ast
    attempts = 0
    changed = True
    while changed and attempts < budget:
        changed = False
        for index in range(len(_main(best)["body"]) - 1, -1, -1):
            attempts += 1
            if attempts > budget:
                break
            candidate = copy.deepcopy(best)
            del _main(candidate)["body"][index]
            if reproduces(candidate) and _ast_size(candidate) < _ast_size(best):
                best = candidate
                changed = True
                break
    return best


def _failure_report(seed, label, ast, args, expected, mismatch,
                    minimized=True):
    order_name, stage, detail = mismatch
    source = ast
    if minimized:
        try:
            source = _minimize(ast, args, expected, order_name, stage)
        except Exception as exc:  # minimization must never mask the failure
            source = ast
            detail += f"\n(minimization itself failed: {type(exc).__name__}: {exc})"
    return (
        f"differential mismatch for {label}\n"
        f"seed                : {seed}\n"
        f"arguments           : {args!r}\n"
        f"expected (return, emits): {expected!r}\n"
        f"pass order          : {order_name}\n"
        f"first disagreement  : stage {stage!r}\n"
        f"detail              : {detail}\n"
        f"minimized source AST:\n"
        f"{json.dumps(source, indent=2, ensure_ascii=False)}"
    )


# ==========================================================================
# Tests: seeded differential + metamorphic sweep
# ==========================================================================


class DifferentialMetamorphicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = generate_corpus(SEED)

    def test_seeded_corpus_covers_required_features(self):
        labels = " ".join(label for label, *_ in self.corpus)
        for feature in (
            "expressions-shadows-branches",
            "bounded-loop-invariant",
            "function-calls",
            "nested-loops",
            "loop-guarded-div",
            "booleans-and-invalid-branch",
        ):
            self.assertIn(feature, labels)
        # Both variants of every template survive generation.
        for name, _builder, variants in _TEMPLATES:
            for variant, _domain in variants:
                self.assertTrue(
                    any(label.startswith(f"{name}#v{variant}")
                        for label, *_ in self.corpus),
                    msg=f"seed {SEED} dropped {name} variant {variant}",
                )

    def test_every_case_matches_through_every_order_and_repetition(self):
        for label, ast, args, expected, exec_once, exec_twice in self.corpus:
            with self.subTest(case=label):
                mismatch = _check_case(
                    ast, args, expected,
                    exec_once=frozenset(exec_once),
                    exec_twice=frozenset(exec_twice))
                if mismatch is not None:
                    self.fail(_failure_report(
                        SEED, label, ast, args, expected, mismatch))

    def test_same_seed_replays_identically(self):
        again = generate_corpus(SEED)
        self.assertEqual(
            [(label, args, expected, exec_once, exec_twice)
             for label, _ast, args, expected, exec_once, exec_twice
             in self.corpus],
            [(label, args, expected, exec_once, exec_twice)
             for label, _ast, args, expected, exec_once, exec_twice
             in again],
        )
        # The concrete ASTs are identical as data, not just by label.
        for (_l, ast_a, *_rest), (_l2, ast_b, *_rest2) in zip(self.corpus,
                                                              again):
            self.assertEqual(json.dumps(ast_a, sort_keys=True),
                             json.dumps(ast_b, sort_keys=True))

    def test_different_seed_changes_the_plan(self):
        # The seed must actually steer enumeration; a second seed changes
        # the visitation order or the subprocess sampling choices.
        other = generate_corpus(SEED + 1)
        plans_a = [
            (label, exec_once, exec_twice)
            for label, _ast, _args, _expected, exec_once, exec_twice
            in self.corpus
        ]
        plans_b = [
            (label, exec_once, exec_twice)
            for label, _ast, _args, _expected, exec_once, exec_twice
            in other
        ]
        self.assertNotEqual(plans_a, plans_b)
        # Feature coverage is still complete under a different seed.
        for name, _builder, variants in _TEMPLATES:
            for variant, _domain in variants:
                self.assertTrue(
                    any(label.startswith(f"{name}#v{variant}")
                        for label, *_ in other))

    def test_licm_before_fold_converges_without_changing_semantics(self):
        # A legal order need not be idempotent after one application; it
        # must nevertheless preserve semantics at every application and
        # reach a stable fixed point after it converges.  LICM-before-fold
        # moves a strictly smaller set on the first pass (its fault fence
        # keeps un-folded loop operands in place); the second application
        # hoists the constants folding exposed, and the third changes
        # nothing.  The test also requires at least one case to exhibit
        # the extra motion, so the convergence assertion is not vacuous.
        order_name, order = LICM_FIRST_ORDER
        saw_extra_motion = False
        plan = [(label, ast, args, expected)
                for label, ast, args, expected, _o, _t in self.corpus]
        # The fixed programs also go through the real process entry:
        # the once- and twice-produced targets must run identically.
        process_plan = []
        for name, builder in FIXED_PROGRAMS:
            ast, rows = builder()
            for args, retv, emits in rows:
                process_plan.append((name, ast, args, (retv, emits)))
        plan.extend((n, a, x, e) for n, a, x, e in process_plan)

        for label, ast, args, expected in plan:
            with self.subTest(case=label):
                baseline = _interpret(_lower(ast), "main", args)
                self.assertIsNone(
                    _outcome_matches_reference(baseline, expected))

                once = _optimize(ast, order)
                twice = optimize_module(once, order)
                thrice = optimize_module(twice, order)

                for stage, module in (("once", once), ("twice", twice),
                                      ("thrice", thrice)):
                    outcome = _interpret(module, "main", args)
                    self.assertEqual(
                        outcome, baseline,
                        msg=f"{label}: semantics changed at stage {stage} "
                            f"of {order_name}")

                text_once, text_twice, text_thrice = (
                    _normalized(once), _normalized(twice),
                    _normalized(thrice))
                if text_once != text_twice:
                    saw_extra_motion = True
                self.assertEqual(
                    text_twice, text_thrice,
                    msg=(f"{label}: {order_name} did not settle on the "
                         "second application"))
                # Every application still recompiles to byte-identical
                # target code for the same configuration.
                self.assertEqual(
                    render_module(twice).encode("utf-8"),
                    render_module(optimize_module(twice, order))
                    .encode("utf-8"))

        for label, ast, args, expected in process_plan:
            once_target = _target(ast, order)
            twice = optimize_module(_optimize(ast, order), order)
            twice_target = render_module(twice).encode("utf-8")
            wanted = RunResult(*_expected_triple(*expected))
            run_once = execute_target(once_target, args)
            run_twice = execute_target(twice_target, args)
            self.assertEqual(
                run_once.triple(), wanted.triple(),
                msg=f"{label}: {order_name} once-target observable mismatch")
            self.assertEqual(
                run_twice.triple(), run_once.triple(),
                msg=(f"{label}: {order_name} targets executed differently "
                     "before and after convergence"))

        self.assertTrue(
            saw_extra_motion,
            msg="at least one program must expose the fence-delayed LICM "
                "motion; otherwise the convergence test is vacuous")

    def test_cross_order_equivalence_is_not_just_text_identity(self):
        # Guard against a vacuous pass: the raw-core order must produce
        # different bytes than the canonical default order for a healthy
        # share of the corpus while running identically.
        differing = 0
        for label, ast, args, expected, _o, _t in self.corpus:
            canonical = _target(ast, DEFAULT_ORDER)
            raw = _target(ast, ORDERS[RAW_CORE_INDEX][1])
            if canonical != raw:
                differing += 1
                run_raw = execute_target(raw, args)
                wanted = RunResult(*_expected_triple(*expected))
                self.assertEqual(
                    run_raw.triple(), wanted.triple(),
                    msg=f"{label}: raw-core target runs differently",
                )
        self.assertGreaterEqual(
            differing, len(self.corpus) // 2,
            msg="raw-core targets should be byte-different for most cases; "
                "a text-identity-only comparison would mask this",
        )


class FixedRegressionTests(unittest.TestCase):
    """Pinned hand-written samples for the three interacting orders and
    the side-effect / non-hoistable / trap guards."""

    def test_fixed_programs_match_pinned_observables(self):
        # The in-process interpreter exhaustively checks every order once
        # and twice; the process entry samples the canonical order and the
        # deliberately raw order, including the second execution.
        sampled = frozenset({0, RAW_CORE_INDEX})
        for name, builder in FIXED_PROGRAMS:
            ast, rows = builder()
            for args, expected_return, expected_emits in rows:
                with self.subTest(sample=name, arguments=args):
                    mismatch = _check_case(
                        ast, args, (expected_return, expected_emits),
                        exec_once=sampled, exec_twice={RAW_CORE_INDEX})
                    if mismatch is not None:
                        self.fail(_failure_report(
                            SEED, name, ast, args,
                            (expected_return, expected_emits), mismatch,
                            minimized=False))

    def test_unused_call_result_is_never_optimized_away(self):
        ast, rows = fold_dce_program()
        for order_name, order in ORDERS:
            module = _optimize(ast, order)
            with self.subTest(order=order_name):
                calls = [
                    ins.name
                    for fn in module.functions if fn.name == "main"
                    for block in fn.blocks
                    for ins in block.instructions if isinstance(ins, Call)
                ]
                # Both emits survive even though "ignored"'s result is
                # dead; deleting it would erase observable stdout.
                self.assertEqual(calls.count("emit"), 2)

    def test_inline_shaped_call_survives_every_order(self):
        ast, _rows = inline_then_fold_program()
        for order_name, order in ORDERS:
            module = _optimize(ast, order)
            with self.subTest(order=order_name):
                main = next(f for f in module.functions if f.name == "main")
                calls = [ins.name for b in main.blocks for ins in b.instructions
                         if isinstance(ins, Call)]
                self.assertEqual(calls, ["dbl", "emit", "emit"])

    def test_loop_constant_hoisted_but_carried_and_div_stay_in_body(self):
        ast, _rows = nonhoistable_loop_program()
        default = _optimize(ast, DEFAULT_ORDER)
        main = next(f for f in default.functions if f.name == "main")
        # The loop body is the unique block containing the guarded div.
        body = next(b for b in main.blocks
                    if any(isinstance(i, BinOp) and i.operator == "div"
                           for i in b.instructions))
        operators = [
            (ins.kind, ins.operator) for ins in body.instructions
            if isinstance(ins, BinOp)
        ]
        # The loop-carried mul (5 * i) and the guarded div stay put.
        self.assertIn(("arith", "mul"), operators)
        self.assertIn(("arith", "div"), operators)
        # The folded invariant Const 5 moved out of the body (LICM hoists
        # constants), so it is not defined inside the loop body.
        body_consts = [i.value for i in body.instructions
                       if isinstance(i, Const)]
        self.assertNotIn(5, body_consts)
        # And it lands in an out-of-loop block instead.
        outside_consts = [
            i.value
            for b in main.blocks if b is not body
            for i in b.instructions if isinstance(i, Const)
        ]
        self.assertIn(5, outside_consts)

    def test_loop_trap_site_and_prefix_trace_are_preserved(self):
        ast, rows = guarded_loop_trap_program()
        for args, _ret, emits in rows:
            baseline = _interpret(_lower(ast), "main", args)
            with self.subTest(arguments=args):
                self.assertEqual(baseline.kind, "normal")
                self.assertEqual(baseline.output,
                                 [("emit", (v,)) for v in emits])
        # n == 3 faults on iteration 2, after the prefix emits 0, 1, 2.
        fault_args = (3,)
        baseline_fault = _interpret(_lower(ast), "main", fault_args)
        self.assertEqual(baseline_fault.kind, "fault")
        self.assertEqual(baseline_fault.category, "division-by-zero")
        self.assertEqual(baseline_fault.site, ("main", "b2", "div", 1))
        self.assertEqual(baseline_fault.output,
                         [("emit", (0,)), ("emit", (1,)), ("emit", (2,))])
        for order_name, order in ORDERS:
            module = _optimize(ast, order)
            with self.subTest(order=order_name):
                outcome = _interpret(module, "main", fault_args)
                self.assertEqual(outcome.kind, "fault")
                self.assertEqual(outcome.category, "division-by-zero")
                self.assertEqual(outcome.site, ("main", "b2", "div", 1))
                self.assertEqual(outcome.output, baseline_fault.output)

    def test_loop_trap_target_process_exit_and_stderr_preserved(self):
        ast, _rows = guarded_loop_trap_program()
        target = _target(ast, DEFAULT_ORDER)
        result = execute_target(target, (3,))
        # Exit 3 is the runner's runtime-fault status; stderr carries the
        # function and operator; stdout holds exactly the prefix emits.
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout, b"0\n1\n2\n")
        self.assertIn(b"main", result.stderr)
        self.assertIn(b"div", result.stderr)


# ==========================================================================
# Fixed diagnostics regressions: same exception type and diagnostic
# location as before optimization existed
# ==========================================================================


class DiagnosticRegressionTests(unittest.TestCase):
    def test_arithmetic_type_error_type_and_message(self):
        bad = program(func("f", [], "int", [
            ret(arith("add", int_(1), bool_(False)))]))
        with self.assertRaises(TypeCheckError) as caught:
            lower_module(bad)
        self.assertEqual(
            str(caught.exception),
            "arithmetic 'add' requires int operands, got int and bool")

    def test_assign_type_error_type_and_message(self):
        bad = program(func("f", [param("x", "int")], "int", [
            let("a", "bool", bool_(True)),
            assign("a", int_(1)),
            ret(var("x")),
        ]))
        with self.assertRaises(TypeCheckError) as caught:
            lower_module(bad)
        self.assertEqual(
            str(caught.exception),
            "assign to 'a': declared bool, value is int")

    def test_undefined_symbol_type_and_message(self):
        bad = program(func("f", [], "int", [ret(var("ghost"))]))
        with self.assertRaises(UndefinedSymbolError) as caught:
            lower_module(bad)
        self.assertEqual(
            str(caught.exception),
            "reference to undeclared variable 'ghost'")

    def test_duplicate_symbol_type_and_message(self):
        bad = program(func("f", [], "int", [
            let("x", "int", int_(1)),
            let("x", "int", int_(2)),
            ret(var("x")),
        ]))
        with self.assertRaises(DuplicateSymbolError) as caught:
            lower_module(bad)
        self.assertEqual(
            str(caught.exception),
            "duplicate declaration of 'x' in the same scope")

    def test_missing_return_type_and_message(self):
        bad = program(func("f", [param("c", "bool")], "int", [
            if_(var("c"), [ret(int_(1))], []),
        ]))
        with self.assertRaises(MissingReturnError) as caught:
            lower_module(bad)
        self.assertEqual(
            str(caught.exception),
            "function 'f': missing return on a reachable path")

    def test_invalid_ast_path_is_preserved(self):
        # A structurally malformed literal (int holding a string) is
        # rejected by AST validation with the same locatable path; the
        # type checker is never reached for it.
        bad = program(func("f", [], "int", [
            ret({"kind": "int", "value": "3"})]))
        with self.assertRaises(InvalidAstError) as caught:
            lower_module(bad)
        self.assertEqual(
            caught.exception.path,
            "module.functions[0].body[0].value.value")

    def test_bool_python_value_in_int_literal_is_invalid_ast(self):
        # The validator explicitly rejects a bool Python value masquerading
        # as an int literal (bool is an int subclass).
        bad = program(func("f", [], "int", [
            ret(arith("add", int_(1), {"kind": "int", "value": True}))]))
        with self.assertRaises(InvalidAstError) as caught:
            lower_module(bad)
        self.assertEqual(
            caught.exception.path,
            "module.functions[0].body[0].value.right.value")

    def test_unknown_statement_kind_path_is_preserved(self):
        bad = {"functions": [{
            "name": "f", "params": [], "ret_type": "void",
            "body": [{"kind": "bogus"}],
        }]}
        with self.assertRaises(InvalidAstError) as caught:
            lower_module(bad)
        self.assertEqual(caught.exception.path,
                         "module.functions[0].body[0]")

    def test_diagnostics_fire_before_any_optimization(self):
        # Every ill-typed case is rejected at the existing public entry; a
        # legal order never gets the chance to touch it.
        bad = program(func("f", [], "int", [
            ret(compare("eq", int_(1), bool_(True)))]))
        with self.assertRaises(TypeCheckError) as caught:
            lower_module(bad)
        self.assertEqual(
            str(caught.exception),
            "comparison 'eq' requires matching types, got int and bool")
        # A fresh attempt through the pipeline entry fails at lowering,
        # before the schedule is ever consulted.
        with self.assertRaises(TypeCheckError):
            optimize_module(_lower(bad), DEFAULT_ORDER)

    def test_void_return_type_error_is_unchanged(self):
        bad = program(func("g", [], "void", [ret(int_(3))]))
        with self.assertRaises(TypeCheckError) as caught:
            lower_module(bad)
        self.assertEqual(
            str(caught.exception),
            "void function 'g' returned int")


if __name__ == "__main__":
    unittest.main()
