"""Semantic-equivalence tests for the non-SSA -> pruned SSA conversion.

The existing tests compare node structure and rendered text.  This module
instead *executes* both flavors of IR and proves that, for one and the same
lowered program, :func:`to_ssa` preserves observable behavior:

* the entry function is invoked with fixed, concrete arguments;
* the non-SSA :class:`Module` returned by ``lower_module`` and the new SSA
  :class:`Module` returned by ``to_ssa`` are each interpreted independently;
* the return value and the ordered trace of function calls
  ``(callee name, positional argument tuple)`` must be identical.

The interpreter is deliberately small and works over both IR flavors:

* non-SSA state is a mutable slot file (parameters / ``let`` variables) plus
  temporary storage, so the multi-written result temporary of a
  short-circuit operation simply holds whichever write the executed path
  performed;
* SSA state is a map of unique :class:`Temp` definitions.  Phi inputs are
  chosen from the block *actually arrived from*, so a loop header reads its
  entry values on the first visit and the current round's values on every
  back edge.

Calls that did not happen (untaken ``if`` branches, unevaluated right
operands of short-circuit ``and``/``or``, zero-trip loops) therefore never
appear in the trace.

Everything here is deterministic and standard-library-only; the only
supported pipeline input is the public dict/list AST (the same subset the
README documents, with int/bool/void and add/sub/mul arithmetic).  Any
interpreter failure -- a missing block target or terminator, a phi without
an entry for the actual predecessor, or a read of an undefined value --
raises ``AssertionError``; a case is never skipped.
"""
import json
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
# IR interpreter (shared by the non-SSA and SSA modules)
# --------------------------------------------------------------------------


# Every lowered program in this file is tiny and terminates quickly; this is
# only a belt-and-braces guard against an accidentally non-terminating case.
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


def _assert(condition: bool, message: str) -> None:
    """Every structural interpreter failure surfaces as AssertionError."""
    if not condition:
        raise AssertionError(message)


class _Interpreter:
    """Execute a lowered :class:`Module` of either SSA flavor.

    A run returns ``(return_value, trace)`` where ``trace`` records, in
    occurrence order, one ``(function_name, (positional arguments))`` tuple
    per executed :class:`Call` instruction (including recursive and
    cross-function calls).  A void entry returns ``None``.
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
        trace: list[tuple[str, tuple]] = []
        value = self._invoke(fn_, arguments, trace)
        return value, trace

    # -- one function activation -------------------------------------------

    def _invoke(self, fn_, arguments, trace: list):
        # SSA: unique Temp definitions; non-SSA: mutable Slots plus Temps
        # (the short-circuit result temporary is written once per executed
        # path, so ordinary overwrite semantics suffice).
        definitions: dict[int, object] = {}
        slots: dict[int, object] = {}
        temps: dict[int, object] = {}

        for parameter, argument in zip(fn_.params, arguments):
            if self.ssa:
                _assert(parameter.temp is not None,
                        f"SSA parameter {parameter.name!r} has no definition")
                definitions[id(parameter.temp)] = argument
            else:
                slots[parameter.slot.id] = argument

        def write(ref, value) -> None:
            if self.ssa:
                _assert(isinstance(ref, Temp) and not isinstance(ref, Slot),
                        f"{fn_.name}: non-Temp definition {ref!r} in SSA IR")
                definitions[id(ref)] = value
            elif isinstance(ref, Slot):
                slots[ref.id] = value
            else:
                temps[id(ref)] = value

        def read(ref):
            if self.ssa:
                _assert(isinstance(ref, Temp),
                        f"{fn_.name}: SSA read of non-Temp value {ref!r}")
                _assert(id(ref) in definitions,
                        f"{fn_.name}: read of undefined SSA value {ref}")
                return definitions[id(ref)]
            if isinstance(ref, Slot):
                _assert(ref.id in slots,
                        f"{fn_.name}: read of undefined slot {ref}")
                return slots[ref.id]
            _assert(id(ref) in temps,
                    f"{fn_.name}: read of undefined temporary {ref}")
            return temps[id(ref)]

        def execute(instruction) -> None:
            if isinstance(instruction, Const):
                write(instruction.dest, instruction.value)
            elif isinstance(instruction, Copy):
                write(instruction.dest, read(instruction.src))
            elif isinstance(instruction, BinOp):
                left = read(instruction.left)
                right = read(instruction.right)
                write(instruction.dest,
                      _apply_binop(instruction, left, right, fn_.name))
            elif isinstance(instruction, Call):
                callee = self._functions.get(instruction.name)
                _assert(callee is not None,
                        f"{fn_.name}: call to unknown function "
                        f"{instruction.name!r}")
                values = tuple(read(operand) for operand in instruction.args)
                _assert(
                    len(values) == len(callee.params),
                    f"call to {instruction.name!r}: arity mismatch at run "
                    f"time ({len(values)} for {len(callee.params)})",
                )
                # Log at the moment the call happens, before the callee's
                # own calls, so the trace follows evaluation order.
                trace.append((instruction.name, values))
                write(instruction.dest,
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
                # All incoming values are read off the *actual* predecessor
                # edge before any phi result is bound, matching phi
                # semantics when header phis reference one another.
                pending: list[tuple[Temp, object]] = []
                for phi in block.phis:
                    _assert(previous is not None,
                            f"{fn_.name}:{block.label}: phi in entry block "
                            "has no predecessor edge")
                    source = phi.entries.get(previous)
                    _assert(source is not None,
                            f"{fn_.name}:{block.label}: phi {phi.dest} has "
                            f"no incoming value from predecessor "
                            f"{previous.label}")
                    pending.append((phi.dest, read(source)))
                for destination, value in pending:
                    definitions[id(destination)] = value
            else:
                _assert(not block.phis,
                        f"{fn_.name}:{block.label}: phi node in non-SSA IR")

            for instruction in block.instructions:
                execute(instruction)

            terminator = block.terminator
            _assert(terminator is not None,
                    f"{fn_.name}:{block.label}: missing terminator")

            if isinstance(terminator, Return):
                if terminator.value is None:
                    return None
                return read(terminator.value)

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
                    f"{fn_.name}:{block.label}: unknown terminator "
                    f"{terminator!r}")


def _apply_binop(instruction: BinOp, left, right, fn_name: str):
    if instruction.kind == "arith":
        # The type checker guarantees plain ints; reject bools explicitly so
        # Python's bool/int overlap cannot hide a lowering mix-up.
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


def _interpret(module: Module, entry: str, arguments):
    return _Interpreter(module).run(entry, arguments)


# --------------------------------------------------------------------------
# Test programs (public AST subset only)
# --------------------------------------------------------------------------


def _identity_function(name: str):
    return func(name, [param("x", "int")], "int", [ret(var("x"))])


def _positive_function():
    return func("positive", [param("x", "int")], "bool",
                [ret(compare("gt", var("x"), int_(0)))])


# A) Nested branches with a missing (default) else.  The merge after the
#    inner if must still run; calls only occur on paths actually taken.
def _paths_program():
    return program(
        func("paths", [param("n", "int")], "int", [
            if_(
                call("is_big", [var("n")]),
                [ret(arith("mul",
                           call("choose", [int_(10)]), int_(2)))],
                [if_(
                    call("is_pos", [var("n")]),
                    [ret(call("choose",
                              [arith("add", var("n"), int_(1))]))],
                )],
            ),
            ret(int_(0)),
        ]),
        func("is_big", [param("x", "int")], "bool",
             [ret(compare("gt", var("x"), int_(100)))]),
        func("is_pos", [param("x", "int")], "bool",
             [ret(compare("gt", var("x"), int_(0)))]),
        _identity_function("choose"),
    )


# B) Loop-carried values: the header phi must read this round's value on
#    each back edge, the entry value on the zero-trip path.
def _sumto_program():
    return program(
        func("sumto", [param("n", "int")], "int", [
            let("acc", "int", int_(0)),
            while_(
                call("positive", [var("n")]),
                [
                    assign("acc", arith(
                        "add", var("acc"),
                        call("addend", [var("n")]))),
                    assign("n", call("decr", [var("n")])),
                ],
            ),
            ret(var("acc")),
        ]),
        _positive_function(),
        _identity_function("addend"),
        func("decr", [param("x", "int")], "int",
             [ret(arith("sub", var("x"), int_(1)))]),
    )


# C) An inner same-name let shadows the outer binding; the outer value is
#    visible again after the block.
def _shadow_program():
    return program(
        func("shadow", [], "int", [
            let("x", "int", call("choose", [int_(4)])),
            block([
                let("x", "int", call("choose", [int_(5)])),
                let("z", "int", call("echo", [var("x")])),
            ]),
            ret(arith("add", var("x"), call("echo", [var("x")]))),
        ]),
        _identity_function("choose"),
        _identity_function("echo"),
    )


# D) Short-circuit and/or: the right operand's call is evaluated only on
#    the path that needs it.
def _logical_program():
    left = call("positive", [var("a")])
    right = call("positive", [var("b")])
    return program(
        func("both", [param("a", "int"), param("b", "int")], "bool",
             [ret(logical("and", left, right))]),
        func("either", [param("a", "int"), param("b", "int")], "bool",
             [ret(logical("or",
                          call("positive", [var("a")]),
                          call("positive", [var("b")])))]),
        _positive_function(),
    )


# E) Cross-function forward calls: callers precede callees in the module.
def _forward_call_program():
    return program(
        func("start", [param("a", "int")], "int",
             [ret(call("middle", [arith("add", var("a"), int_(1))]))]),
        func("middle", [param("x", "int")], "int",
             [ret(call("last", [arith("mul", var("x"), int_(2))]))]),
        func("last", [param("y", "int")], "int",
             [ret(arith("sub", var("y"), int_(5)))]),
    )


# F) One variable takes different definitions in the two branches and
#    converges at the merge; each edge must supply its own phi input.
def _magnitude_program():
    return program(
        func("magnitude", [param("n", "int")], "int", [
            let("v", "int", int_(0)),
            if_(
                call("positive", [var("n")]),
                [assign("v", call("negate", [var("n")]))],
                [assign("v", call("as_is", [var("n")]))],
            ),
            ret(arith("add", var("v"), var("v"))),
        ]),
        _positive_function(),
        func("negate", [param("x", "int")], "int",
             [ret(arith("sub", int_(0), var("x")))]),
        _identity_function("as_is"),
    )


# G) Integrated stress case: short-circuit loop condition, an if inside the
#    body that updates carried values on both edges, and a zero-trip path.
def _loop_mix_program():
    return program(
        func("loopmix", [param("n", "int")], "int", [
            let("total", "int", int_(0)),
            while_(
                logical("and",
                        call("positive", [var("n")]),
                        call("cap", [var("n")])),
                [if_(
                    compare("gt", var("n"), int_(5)),
                    [
                        assign("total", arith(
                            "add", var("total"),
                            call("big", [var("n")]))),
                        assign("n", call("decr", [var("n")])),
                    ],
                    [
                        assign("total", arith(
                            "add", var("total"),
                            call("small", [var("n")]))),
                        assign("n", arith("sub", var("n"), int_(1))),
                    ],
                )],
            ),
            ret(var("total")),
        ]),
        _positive_function(),
        func("cap", [param("x", "int")], "bool",
             [ret(compare("lt", var("x"), int_(10)))]),
        _identity_function("big"),
        _identity_function("small"),
        func("decr", [param("x", "int")], "int",
             [ret(arith("sub", var("x"), int_(1)))]),
    )


# H) A void entry function: calls happen through let initializers on
#    branch-specific paths, then the function falls through bare.
def _void_logger_program():
    return program(
        func("logger", [param("n", "int")], "void", [
            if_(
                call("positive", [var("n")]),
                [let("r", "int", call("record", [var("n")]))],
                [let("s", "int", call("record", [int_(0)]))],
            ),
        ]),
        _positive_function(),
        _identity_function("record"),
    )


# (case name, AST, entry function, (arguments, expected return, expected
#  trace)); expected values pin the ground truth in addition to the
#  non-SSA/SSA cross-check.
_CASES = [
    (
        "nested branches with missing else",
        _paths_program(), "paths",
        [
            ((150,), 20, [("is_big", (150,)), ("choose", (10,))]),
            ((5,), 6,
             [("is_big", (5,)), ("is_pos", (5,)), ("choose", (6,))]),
            ((0,), 0, [("is_big", (0,)), ("is_pos", (0,))]),
            ((-3,), 0, [("is_big", (-3,)), ("is_pos", (-3,))]),
        ],
    ),
    (
        "loop-carried values",
        _sumto_program(), "sumto",
        [
            ((0,), 0, [("positive", (0,))]),
            ((3,), 6, [
                ("positive", (3,)), ("addend", (3,)), ("decr", (3,)),
                ("positive", (2,)), ("addend", (2,)), ("decr", (2,)),
                ("positive", (1,)), ("addend", (1,)), ("decr", (1,)),
                ("positive", (0,)),
            ]),
        ],
    ),
    (
        "inner shadowing",
        _shadow_program(), "shadow",
        [
            ((), 8, [
                ("choose", (4,)), ("choose", (5,)),
                ("echo", (5,)), ("echo", (4,)),
            ]),
        ],
    ),
    (
        "short-circuit and",
        _logical_program(), "both",
        [
            ((5, -2), False,
             [("positive", (5,)), ("positive", (-2,))]),
            ((0, 9), False, [("positive", (0,))]),
        ],
    ),
    (
        "short-circuit or",
        _logical_program(), "either",
        [
            ((-1, 8), True,
             [("positive", (-1,)), ("positive", (8,))]),
            ((3, -1), True, [("positive", (3,))]),
            ((-1, -2), False,
             [("positive", (-1,)), ("positive", (-2,))]),
        ],
    ),
    (
        "forward calls",
        _forward_call_program(), "start",
        [
            ((10,), 17, [("middle", (11,)), ("last", (22,))]),
            ((-3,), -9, [("middle", (-2,)), ("last", (-4,))]),
        ],
    ),
    (
        "branch-convergent definitions",
        _magnitude_program(), "magnitude",
        [
            ((4,), -8, [("positive", (4,)), ("negate", (4,))]),
            ((-7,), -14, [("positive", (-7,)), ("as_is", (-7,))]),
        ],
    ),
    (
        "loop with inner branch and short-circuit condition",
        _loop_mix_program(), "loopmix",
        [
            ((0,), 0, [("positive", (0,))]),
            ((10,), 0, [("positive", (10,)), ("cap", (10,))]),
            ((2,), 3, [
                ("positive", (2,)), ("cap", (2,)), ("small", (2,)),
                ("positive", (1,)), ("cap", (1,)), ("small", (1,)),
                ("positive", (0,)),
            ]),
            ((7,), 28, [
                ("positive", (7,)), ("cap", (7,)),
                ("big", (7,)), ("decr", (7,)),
                ("positive", (6,)), ("cap", (6,)),
                ("big", (6,)), ("decr", (6,)),
                ("positive", (5,)), ("cap", (5,)), ("small", (5,)),
                ("positive", (4,)), ("cap", (4,)), ("small", (4,)),
                ("positive", (3,)), ("cap", (3,)), ("small", (3,)),
                ("positive", (2,)), ("cap", (2,)), ("small", (2,)),
                ("positive", (1,)), ("cap", (1,)), ("small", (1,)),
                ("positive", (0,)),
            ]),
        ],
    ),
    (
        "void entry with branch-local calls",
        _void_logger_program(), "logger",
        [
            ((3,), None, [("positive", (3,)), ("record", (3,))]),
            ((-1,), None, [("positive", (-1,)), ("record", (0,))]),
        ],
    ),
]


# Literal-bool branch (the other cases derive booleans from compares and
# calls); also exercises the interpreter's literal Const path.
_BOOL_LITERAL_AST = program(
    func("lit", [], "bool", [
        if_(bool_(True), [ret(bool_(True))], [ret(bool_(False))]),
    ])
)


def _failure_message(name, ast, entry, arguments, non_ssa_result,
                     ssa_result) -> str:
    return (
        f"semantic mismatch in case {name!r}, entry {entry!r}, "
        f"arguments {arguments!r}\n"
        f"reproducible AST:\n"
        f"{json.dumps(ast, indent=2, ensure_ascii=False)}\n"
        f"non-SSA result (return value, call trace): {non_ssa_result!r}\n"
        f"SSA     result (return value, call trace): {ssa_result!r}"
    )


class SemanticEquivalenceTests(unittest.TestCase):
    def test_non_ssa_and_ssa_modules_agree(self):
        for name, ast, entry, runs in _CASES:
            for arguments, expected_return, expected_trace in runs:
                with self.subTest(case=name, arguments=arguments):
                    module = lower_module(ast)
                    non_ssa_result = _interpret(module, entry, arguments)

                    ssa_module = to_ssa(module)
                    ssa_result = _interpret(ssa_module, entry, arguments)

                    self.assertEqual(
                        non_ssa_result, ssa_result,
                        msg=_failure_message(
                            name, ast, entry, arguments,
                            non_ssa_result, ssa_result),
                    )
                    # Ground-truth pin: agreement between the two forms is
                    # necessary but not sufficient, so anchor the observable
                    # result independently.
                    self.assertEqual(
                        ssa_result, (expected_return, expected_trace),
                        msg=(
                            f"case {name!r} arguments {arguments!r}: "
                            f"unexpected observable result {ssa_result!r}"
                        ),
                    )

    def test_bool_literal_branches_agree(self):
        # Direct literal-true branch through the interpreter's bool path.
        module = lower_module(_BOOL_LITERAL_AST)
        non_ssa_result = _interpret(module, "lit", ())
        ssa_result = _interpret(to_ssa(module), "lit", ())
        self.assertEqual(non_ssa_result, (True, []))
        self.assertEqual(ssa_result, (True, []))

    def test_results_are_deterministic_on_repeated_runs(self):
        def one_pass():
            observed = []
            for name, ast, entry, runs in _CASES:
                for arguments, _expected_return, _expected_trace in runs:
                    module = lower_module(ast)
                    non_ssa_result = _interpret(module, entry, arguments)
                    ssa_result = _interpret(
                        to_ssa(module), entry, arguments)
                    observed.append(
                        (name, arguments, non_ssa_result, ssa_result))
            return observed

        first = one_pass()
        second = one_pass()
        # Re-running the whole deterministic table must reproduce both the
        # exact results and the fixed _CASES (case, arguments) iteration
        # order.
        self.assertEqual(first, second)

    def test_trace_omits_calls_on_unexecuted_paths(self):
        # Focused assertions for the three "call must not happen" contexts:
        # missing-else fallthrough, short-circuit right operand, zero-trip
        # loop body.
        _, paths_ast, _, _ = _CASES[0]
        result = _interpret(to_ssa(lower_module(paths_ast)), "paths", (0,))
        self.assertEqual(result[1], [("is_big", (0,)), ("is_pos", (0,))])

        logical_ast = _logical_program()
        result = _interpret(
            to_ssa(lower_module(logical_ast)), "both", (0, 9))
        self.assertEqual(result, (False, [("positive", (0,))]))

        _, sumto_ast, _, _ = _CASES[1]
        result = _interpret(
            to_ssa(lower_module(sumto_ast)), "sumto", (0,))
        self.assertEqual(result, (0, [("positive", (0,))]))


if __name__ == "__main__":
    unittest.main()
