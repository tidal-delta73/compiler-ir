"""Semantic equivalence tests: non-SSA and SSA modules compute the same thing.

The structural tests in ``test_ssa`` compare node shapes and rendered text.
These tests go further: a small tree-walking interpreter executes the
``Module`` returned by :func:`lower_module` and the new ``Module`` returned
by :func:`to_ssa` on the same deterministic entry arguments, and both the
return value and the function-call trace (callee name plus arguments, in
order of occurrence) must agree exactly.  This is the regression baseline
for later optimisation passes (constant propagation, dead-code elimination,
pass-ordering combinations): any pass that preserves this file's
before/after equality preserves observable program behaviour.

Only the public AST subset from the README is used (``int``/``bool``/``void``
types, ``add``/``sub``/``mul`` arithmetic so no unspecified division
semantics sneak in), and every program plus argument set is statically
guaranteed to terminate.  All data is deterministic and standard-library
only, so repeated runs produce identical case order and results.

Interpreter contract: control flow follows the actually executed path --
phi inputs are selected by the real predecessor (loop back edges read the
current iteration's value), so calls in untaken ``if`` branches, in
short-circuited right operands, or in zero-iteration loop bodies never
appear in the trace.  Any structural defect -- a terminator targeting a
block outside the function, a missing terminator, a phi with no entry for
the actual predecessor, or a read of an undefined value -- raises
``AssertionError`` and fails the test instead of skipping it.
"""
import unittest

from compiler_ir import (
    BinOp,
    Block,
    Branch,
    Call,
    Const,
    Copy,
    Jump,
    Return,
    Temp,
    lower_module,
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


# --------------------------------------------------------------------------
# Interpreter for both module flavors
# --------------------------------------------------------------------------

# Safety net only: every case is terminating by construction, so hitting
# this bound means the IR (or the interpreter) is broken, not the test data.
_STEP_LIMIT = 100_000


def _apply_binop(kind, operator, left, right):
    if kind == "arith":
        if operator == "add":
            return left + right
        if operator == "sub":
            return left - right
        if operator == "mul":
            return left * right
        # div/mod are deliberately not interpreted: the public semantics
        # (e.g. division by zero) are not pinned down, and no test case
        # uses them.
        raise AssertionError(f"unsupported arithmetic operator {operator!r}")
    if kind == "compare":
        if operator == "eq":
            return left == right
        if operator == "ne":
            return left != right
        if operator == "lt":
            return left < right
        if operator == "le":
            return left <= right
        if operator == "gt":
            return left > right
        if operator == "ge":
            return left >= right
        raise AssertionError(f"unsupported comparison operator {operator!r}")
    raise AssertionError(f"unsupported binop kind {kind!r}")


class _Interpreter:
    """Executes one module flavor, recording every call in ``self.trace``.

    The trace holds ``(callee_name, argument_tuple)`` entries in order of
    occurrence.  Values are plain Python ``int``/``bool``; ``void`` returns
    surface as ``None``.
    """

    def __init__(self, module):
        self.functions = {f.name: f for f in module.functions}
        self.trace = []

    def invoke(self, name, args):
        func = self.functions.get(name)
        if func is None:
            raise AssertionError(f"unknown entry function {name!r}")
        return self._call(func, tuple(args))

    @staticmethod
    def _read(env, ref):
        try:
            return env[ref]
        except KeyError:
            raise AssertionError(f"read of undefined value {ref}") from None

    def _call(self, func, args):
        if len(args) != len(func.params):
            raise AssertionError(
                f"{func.name} expects {len(func.params)} args, got {len(args)}"
            )
        env = {}
        for parameter, argument in zip(func.params, args):
            key = parameter.temp if func.ssa else parameter.slot
            if key is None:
                raise AssertionError(
                    f"parameter {parameter.name!r} of {func.name} has no "
                    "value cell for this module flavor"
                )
            env[key] = argument

        blocks = set(func.blocks)
        block = func.entry
        predecessor = None
        for _ in range(_STEP_LIMIT):
            if block not in blocks:
                raise AssertionError(
                    f"terminator of {predecessor.label if predecessor else 'entry'} "
                    f"targets a block that is not part of {func.name}"
                )
            # Phi inputs are chosen by the predecessor actually arrived
            # from; on a loop back edge this is the body block of the
            # current iteration, so the backedge value is this round's.
            for phi in block.phis:
                if predecessor not in phi.entries:
                    raise AssertionError(
                        f"phi {phi.dest} in {func.name}.{block.label} has no "
                        f"incoming value for predecessor "
                        f"{predecessor.label if predecessor else None}"
                    )
                env[phi.dest] = self._read(env, phi.entries[predecessor])
            for ins in block.instructions:
                self._execute(env, ins)
            term = block.terminator
            if term is None:
                raise AssertionError(
                    f"block {func.name}.{block.label} has no terminator"
                )
            if isinstance(term, Return):
                if term.value is None:
                    return None
                return self._read(env, term.value)
            if isinstance(term, Jump):
                predecessor, block = block, term.target
                continue
            if isinstance(term, Branch):
                cond = self._read(env, term.cond)
                predecessor = block
                block = term.true_target if cond else term.false_target
                continue
            raise AssertionError(f"unknown terminator {term!r}")
        raise AssertionError(
            f"{func.name} did not terminate within {_STEP_LIMIT} block hops"
        )

    def _execute(self, env, ins):
        if isinstance(ins, Const):
            env[ins.dest] = ins.value
            return
        if isinstance(ins, Copy):
            env[ins.dest] = self._read(env, ins.src)
            return
        if isinstance(ins, BinOp):
            env[ins.dest] = _apply_binop(
                ins.kind,
                ins.operator,
                self._read(env, ins.left),
                self._read(env, ins.right),
            )
            return
        if isinstance(ins, Call):
            callee = self.functions.get(ins.name)
            if callee is None:
                raise AssertionError(f"call to unknown function {ins.name!r}")
            args = tuple(self._read(env, arg) for arg in ins.args)
            self.trace.append((ins.name, args))
            env[ins.dest] = self._call(callee, args)
            return
        raise AssertionError(f"unknown instruction {ins!r}")


def interpret(module, entry, args):
    """Run ``entry(*args)`` and return ``(return_value, call_trace)``."""
    interpreter = _Interpreter(module)
    value = interpreter.invoke(entry, args)
    return value, interpreter.trace


# --------------------------------------------------------------------------
# Deterministic test programs (README AST subset, add/sub/mul only)
# --------------------------------------------------------------------------


def nested_if_default_else():
    """Nested branch whose inner ``if`` has no else (the default else)."""
    return program(func(
        "f",
        [param("c", "bool"), param("d", "bool"), param("x", "int")],
        "int",
        [
            let("a", "int", int_(0)),
            if_(
                var("c"),
                [if_(var("d"), [assign("a", var("x"))])],
                [assign("a", int_(7))],
            ),
            ret(var("a")),
        ],
    ))


def loop_carried_values():
    """Loop-carried accumulator fed by a call inside the loop body."""
    return program(
        func(
            "double", [param("x", "int")], "int",
            [ret(arith("mul", var("x"), int_(2)))],
        ),
        func(
            "accumulate", [param("n", "int")], "int",
            [
                let("acc", "int", int_(1)),
                while_(
                    compare("gt", var("n"), int_(0)),
                    [
                        assign(
                            "acc",
                            arith("add", var("acc"),
                                  call("double", [var("n")])),
                        ),
                        assign("n", arith("sub", var("n"), int_(1))),
                    ],
                ),
                ret(var("acc")),
            ],
        ),
    )


def shadowing_program():
    """An inner ``let`` with the same name shadows the parameter."""
    return program(func(
        "f", [param("x", "int")], "int",
        [
            let("y", "int", arith("add", var("x"), int_(1))),
            block([
                let("x", "int", int_(100)),
                assign("y", arith("add", var("y"), var("x"))),
            ]),
            ret(var("y")),
        ],
    ))


def short_circuit_program():
    """Calls in the right operand of ``and``/``or`` run only when reached."""
    return program(
        func(
            "positive", [param("x", "int")], "bool",
            [ret(compare("gt", var("x"), int_(0)))],
        ),
        func(
            "f", [param("a", "bool"), param("x", "int")], "bool",
            [
                let("r", "bool",
                    logical("and", var("a"), call("positive", [var("x")]))),
                let("s", "bool",
                    logical("or", var("a"), call("positive", [var("x")]))),
                ret(logical("or", var("r"), var("s"))),
            ],
        ),
    )


def branch_merge_calls():
    """One variable, a different call-defined value per branch, then merged."""
    return program(
        func(
            "inc", [param("x", "int")], "int",
            [ret(arith("add", var("x"), int_(1)))],
        ),
        func(
            "dec", [param("x", "int")], "int",
            [ret(arith("sub", var("x"), int_(1)))],
        ),
        func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", int_(0)),
                if_(
                    var("c"),
                    [assign("a", call("inc", [var("x")]))],
                    [assign("a", call("dec", [var("x")]))],
                ),
                ret(arith("mul", var("a"), int_(2))),
            ],
        ),
    )


def forward_call_program():
    """The callee is declared after the caller (global function table)."""
    return program(
        func(
            "use", [param("x", "int")], "int",
            [ret(arith("add", call("later", [var("x")]), int_(1)))],
        ),
        func(
            "later", [param("x", "int")], "int",
            [ret(arith("mul", var("x"), int_(3)))],
        ),
    )


# (name, ast_builder, entry, [(args, expected_value, expected_trace), ...])
CASES = [
    (
        "nested_if_default_else",
        nested_if_default_else,
        "f",
        [
            ((True, True, 5), 5, []),
            ((True, False, 5), 0, []),   # inner default else keeps a = 0
            ((False, True, 9), 7, []),
        ],
    ),
    (
        "loop_carried_values",
        loop_carried_values,
        "accumulate",
        [
            # Zero iterations: the in-loop call must not appear in the trace.
            ((0,), 1, []),
            ((3,), 13, [("double", (3,)), ("double", (2,)), ("double", (1,))]),
        ],
    ),
    (
        "shadowing",
        shadowing_program,
        "f",
        [
            ((5,), 106, []),   # y = 5 + 1, then y += inner x (100)
            ((-1,), 100, []),
        ],
    ),
    (
        "short_circuit",
        short_circuit_program,
        "f",
        [
            # and evaluates the right operand; or short-circuits it away.
            ((True, 3), True, [("positive", (3,))]),
            # and short-circuits; or evaluates the right operand.
            ((False, 3), True, [("positive", (3,))]),
            ((False, -2), False, [("positive", (-2,))]),
            ((True, -2), True, [("positive", (-2,))]),
        ],
    ),
    (
        "branch_merge_calls",
        branch_merge_calls,
        "f",
        [
            # Only the taken branch's call may appear in the trace.
            ((True, 4), 10, [("inc", (4,))]),
            ((False, 4), 6, [("dec", (4,))]),
        ],
    ),
    (
        "forward_call",
        forward_call_program,
        "use",
        [
            ((2,), 7, [("later", (2,))]),
            ((0,), 1, [("later", (0,))]),
        ],
    ),
]


def _run_case(builder, entry, runs):
    """Interpret both module flavors for every argument set of a case."""
    ast = builder()
    module = lower_module(ast)
    ssa_module = to_ssa(module)
    results = []
    for args, _expected_value, _expected_trace in runs:
        results.append((
            interpret(module, entry, args),
            interpret(ssa_module, entry, args),
        ))
    return ast, results


# --------------------------------------------------------------------------
# Equivalence tests
# --------------------------------------------------------------------------


class SemanticEquivalenceTests(unittest.TestCase):
    def test_non_ssa_and_ssa_produce_identical_results(self):
        for name, builder, entry, runs in CASES:
            ast = builder()
            module = lower_module(ast)
            ssa_module = to_ssa(module)
            for args, expected_value, expected_trace in runs:
                with self.subTest(case=name, args=args):
                    non_ssa = interpret(module, entry, args)
                    ssa = interpret(ssa_module, entry, args)
                    detail = (
                        f"\ncase: {name}"
                        f"\nAST: {ast!r}"
                        f"\nentry: {entry}{args!r}"
                        f"\nnon-SSA result: {non_ssa!r}"
                        f"\nSSA result:     {ssa!r}"
                    )
                    self.assertEqual(
                        non_ssa,
                        ssa,
                        "non-SSA and SSA disagree" + detail,
                    )
                    # Both flavors also match the independently computed
                    # expectation, so the call-trace guarantees (no calls
                    # from untaken branches, short-circuited operands or
                    # zero-iteration loop bodies) are checked directly.
                    self.assertEqual(
                        non_ssa,
                        (expected_value, expected_trace),
                        "unexpected observable result" + detail,
                    )

    def test_results_are_repeatable_across_runs(self):
        first = [_run_case(builder, entry, runs)
                 for _name, builder, entry, runs in CASES]
        second = [_run_case(builder, entry, runs)
                  for _name, builder, entry, runs in CASES]
        self.assertEqual(
            [results for _ast, results in first],
            [results for _ast, results in second],
        )


# --------------------------------------------------------------------------
# Interpreter failure contract
# --------------------------------------------------------------------------


class InterpreterFailureTests(unittest.TestCase):
    """Structural defects fail with AssertionError instead of being skipped."""

    def _module(self):
        return lower_module(branch_merge_calls())

    def test_missing_terminator(self):
        module = self._module()
        module.functions[2].blocks[0].terminator = None
        with self.assertRaises(AssertionError):
            interpret(module, "f", (True, 4))

    def test_unknown_block_target(self):
        module = self._module()
        branch = module.functions[2].blocks[0].terminator
        assert isinstance(branch, Branch)
        branch.true_target = Block(999)  # not part of the function
        with self.assertRaises(AssertionError):
            interpret(module, "f", (True, 4))

    def test_phi_without_entry_for_actual_predecessor(self):
        ssa_module = to_ssa(self._module())
        func_f = ssa_module.functions[2]
        phi = next(p for b in func_f.blocks for p in b.phis)
        phi.entries.pop(next(iter(phi.entries)))
        with self.assertRaises(AssertionError):
            interpret(ssa_module, "f", (True, 4))

    def test_read_of_undefined_value(self):
        module = lower_module(
            program(func("f", [], "int", [ret(int_(1))]))
        )
        module.functions[0].blocks[0].terminator = Return(Temp(99, "int"))
        with self.assertRaises(AssertionError):
            interpret(module, "f", ())


if __name__ == "__main__":
    unittest.main()
