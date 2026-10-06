"""Tests for SSA constant folding and propagation (:func:`fold_constants`).

The pass is tested six ways:

* the API contract -- new SSA module, input never mutated, ``TypeError``
  for non-modules, ``ValueError`` for non-SSA modules, independent copies
  for the empty module and modules with nothing to fold;
* folding -- every arithmetic and comparison operator folds on known
  operands, folded literals keep their int/bool type, chains propagate,
  and truncated ``div``/``mod`` semantics match the runtime;
* trapping operations -- a ``div``/``mod`` with a known zero divisor keeps
  its ``BinOp`` verbatim, and ``Call`` results stay unknown with the call
  order untouched;
* phi folding -- a phi becomes a ``Const`` of the same SSA value only when
  every reachable predecessor supplies the same known literal;
* preservation -- function/block/terminator layout, unfolded instruction
  order and SSA numbers are untouched, and the output is SSA-closed;
* semantics -- the shared IR interpreters observe the same return values,
  call traces and runtime faults as the input, also after a trailing
  :func:`eliminate_dead_code` cleanup.
"""
import unittest

from compiler_ir import (
    BinOp,
    Branch,
    Call,
    Const,
    Module,
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
from test_dce import _all_defs, _container_objects, assert_ssa_closed
from test_semantic_equivalence import (
    _CASES as _SEMANTIC_CASES,
    _interpret as _interpret_plain,
)
from test_pass_ordering import (
    _CASES as _ORDER_CASES,
    ZERO_DIVISOR,
    _interpret as _interpret_outcome,
)


# --------------------------------------------------------------------------
# Small builders
# --------------------------------------------------------------------------


def _fold(ast):
    return fold_constants(to_ssa(lower_module(ast)))


def _single_arith(op, left, right):
    """f() -> int { return <left> op <right> } folded once."""
    return _fold(program(func(
        "f", [], "int", [ret(arith(op, int_(left), int_(right)))]
    )))


def _instructions(fn_):
    return [ins for block in fn_.blocks for ins in block.instructions]


def _echo_program(body):
    return program(
        func("f", [], "int", body),
        func("echo", [param("x", "int")], "int", [ret(var("x"))]),
    )


# --------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------


def _foldable_ast():
    return program(func(
        "f", [param("x", "int")], "int",
        [ret(arith("add", arith("add", int_(1), int_(2)), var("x")))],
    ))


class ApiTests(unittest.TestCase):
    def test_returns_new_module_marked_ssa(self):
        ssa = to_ssa(lower_module(_foldable_ast()))
        result = fold_constants(ssa)
        self.assertIsInstance(result, Module)
        self.assertIsNot(result, ssa)
        self.assertTrue(result.ssa)
        self.assertTrue(all(f.ssa for f in result.functions))

    def test_input_module_is_not_mutated(self):
        ssa = to_ssa(lower_module(_foldable_ast()))
        before = render_module(ssa)
        fold_constants(ssa)
        self.assertEqual(render_module(ssa), before)

    def test_shares_no_mutable_containers_with_input(self):
        ssa = to_ssa(lower_module(_foldable_ast()))
        result = fold_constants(ssa)
        self.assertTrue(
            _container_objects(ssa).isdisjoint(_container_objects(result))
        )

    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.assertRaises(TypeError):
                fold_constants(bad)

    def test_non_ssa_module_raises_value_error(self):
        non_ssa = lower_module(_foldable_ast())
        self.assertFalse(non_ssa.ssa)
        with self.assertRaises(ValueError):
            fold_constants(non_ssa)
        # An explicitly empty non-SSA module is still the wrong flavor.
        with self.assertRaises(ValueError):
            fold_constants(Module([], ssa=False))

    def test_empty_module_returns_independent_copy(self):
        empty = to_ssa(lower_module(program()))
        self.assertEqual(empty.functions, [])
        result = fold_constants(empty)
        self.assertIsNot(result, empty)
        self.assertTrue(result.ssa)
        self.assertEqual(result.functions, [])
        self.assertEqual(render_module(result), render_module(empty))

    def test_nothing_to_fold_returns_independent_equivalent_copy(self):
        # Parameters and call results are never known; the only Const is
        # already a literal, so no definition can be folded.
        ssa = to_ssa(lower_module(_echo_program([
            let("v", "int", call("echo", [int_(5)])),
            ret(arith("add", var("v"), var("v"))),
        ])))
        once = fold_constants(ssa)
        self.assertEqual(render_module(once), render_module(ssa))
        self.assertTrue(
            _container_objects(ssa).isdisjoint(_container_objects(once))
        )
        twice = fold_constants(once)
        self.assertIsNot(twice, once)
        self.assertEqual(render_module(twice), render_module(once))


# --------------------------------------------------------------------------
# Arithmetic and comparison folding
# --------------------------------------------------------------------------


class ArithFoldingTests(unittest.TestCase):
    def _folded_const(self, op, left, right):
        fn_ = _single_arith(op, left, right).functions[0]
        block = fn_.blocks[0]
        self.assertFalse(
            any(isinstance(i, BinOp) for i in block.instructions),
            f"{op} {left} {right}: BinOp survived folding",
        )
        consts = [i for i in block.instructions if isinstance(i, Const)]
        self.assertIs(block.terminator.value, consts[-1].dest)
        return consts[-1]

    def test_add_sub_mul_fold(self):
        for op, left, right, expected in (
            ("add", 1, 2, 3),
            ("add", -5, 5, 0),
            ("sub", 7, 10, -3),
            ("mul", 6, 7, 42),
            ("mul", -3, -4, 12),
        ):
            with self.subTest(op=op, left=left, right=right):
                const = self._folded_const(op, left, right)
                self.assertEqual(const.value, expected)
                self.assertEqual(const.dest.type, "int")

    def test_div_mod_truncate_toward_zero(self):
        for op, left, right, expected in (
            ("div", 7, 2, 3),
            ("div", -7, 2, -3),
            ("div", 7, -2, -3),
            ("div", -7, -2, 3),
            ("mod", 7, 2, 1),
            ("mod", -7, 2, -1),
            ("mod", 7, -2, 1),
            ("mod", -7, -2, -1),
        ):
            with self.subTest(op=op, left=left, right=right):
                const = self._folded_const(op, left, right)
                self.assertEqual(const.value, expected)
                # The language invariant: a == div(a, b) * b + mod(a, b).
                div = self._folded_const("div", left, right).value
                mod = self._folded_const("mod", left, right).value
                self.assertEqual(left, div * right + mod)

    def test_comparisons_fold_to_bool_consts(self):
        for op, left, right, expected in (
            ("eq", 3, 3, True),
            ("eq", 3, 4, False),
            ("ne", 3, 4, True),
            ("lt", 3, 4, True),
            ("le", 4, 4, True),
            ("gt", 4, 3, True),
            ("ge", 3, 4, False),
        ):
            with self.subTest(op=op, left=left, right=right):
                fn_ = _fold(program(func(
                    "f", [], "bool",
                    [ret(compare(op, int_(left), int_(right)))],
                ))).functions[0]
                block = fn_.blocks[0]
                self.assertFalse(any(
                    isinstance(i, BinOp) for i in block.instructions
                ))
                const = block.instructions[-1]
                self.assertIsInstance(const, Const)
                self.assertIs(block.terminator.value, const.dest)
                self.assertEqual(const.value, expected)
                self.assertIs(type(const.value), bool)
                self.assertEqual(const.dest.type, "bool")

    def test_bool_literals_fold_through_comparison(self):
        fn_ = _fold(program(func(
            "f", [], "bool", [ret(compare("eq", bool_(True), bool_(False)))],
        ))).functions[0]
        const = fn_.blocks[0].instructions[-1]
        self.assertIsInstance(const, Const)
        self.assertIs(const.value, False)

    def test_folded_results_propagate_along_chains(self):
        # (1 + 2) * 3 folds all the way to a single literal 9.
        fn_ = _fold(program(func(
            "f", [], "int",
            [ret(arith("mul", arith("add", int_(1), int_(2)), int_(3)))],
        ))).functions[0]
        block = fn_.blocks[0]
        self.assertFalse(any(isinstance(i, BinOp) for i in block.instructions))
        const = block.instructions[-1]
        self.assertEqual(const.value, 9)
        self.assertIs(block.terminator.value, const.dest)

    def test_unknown_operand_keeps_binop(self):
        fn_ = _fold(program(func(
            "f", [param("x", "int")], "int",
            [ret(arith("add", var("x"), int_(1)))],
        ))).functions[0]
        binops = [i for i in _instructions(fn_) if isinstance(i, BinOp)]
        self.assertEqual(len(binops), 1)
        self.assertEqual(binops[0].operator, "add")


# --------------------------------------------------------------------------
# Trapping operations and calls
# --------------------------------------------------------------------------


class TrappingOperationTests(unittest.TestCase):
    def test_literal_zero_divisor_keeps_binop(self):
        for op in ("div", "mod"):
            with self.subTest(op=op):
                fn_ = _single_arith(op, 7, 0).functions[0]
                binops = [i for i in _instructions(fn_)
                          if isinstance(i, BinOp)]
                self.assertEqual(len(binops), 1)
                self.assertEqual(binops[0].operator, op)
                self.assertIs(
                    fn_.blocks[0].terminator.value, binops[0].dest
                )

    def test_propagated_zero_divisor_keeps_binop(self):
        # The zero arrives through a folded `let`, not a literal operand.
        for op in ("div", "mod"):
            with self.subTest(op=op):
                fn_ = _fold(program(func(
                    "f", [param("x", "int")], "int",
                    [
                        let("z", "int", int_(0)),
                        ret(arith(op, var("x"), var("z"))),
                    ],
                ))).functions[0]
                binops = [i for i in _instructions(fn_)
                          if isinstance(i, BinOp)]
                self.assertEqual(len(binops), 1)
                self.assertEqual(binops[0].operator, op)

    def test_nonzero_known_divisor_still_folds(self):
        const = ArithFoldingTests._folded_const(self, "div", 9, 2)
        self.assertEqual(const.value, 4)

    def test_call_result_is_unknown_and_call_order_preserved(self):
        fn_ = _fold(_echo_program([
            let("a", "int", call("echo", [int_(1)])),
            let("b", "int", call("echo", [int_(2)])),
            ret(arith("add", var("a"), var("b"))),
        ])).functions[0]
        calls = [i for i in _instructions(fn_) if isinstance(i, Call)]
        self.assertEqual([c.name for c in calls], ["echo", "echo"])
        # The add of two call results cannot fold.
        binops = [i for i in _instructions(fn_) if isinstance(i, BinOp)]
        self.assertEqual(len(binops), 1)
        self.assertEqual(binops[0].operator, "add")
        # Both call arguments folded to their literal Const definitions.
        for call_ins, literal in zip(calls, (1, 2)):
            owner = next(
                i for i in _instructions(fn_)
                if isinstance(i, Const) and i.dest is call_ins.args[0]
            )
            self.assertEqual(owner.value, literal)


# --------------------------------------------------------------------------
# Phi folding
# --------------------------------------------------------------------------


def _merge_program(then_value, else_value):
    # `a` takes a (possibly equal) literal on each branch and is returned.
    return program(func(
        "f", [param("c", "bool")], "int",
        [
            let("a", "int", int_(0)),
            if_(var("c"),
                [assign("a", int_(then_value))],
                [assign("a", int_(else_value))]),
            ret(var("a")),
        ],
    ))


class PhiFoldingTests(unittest.TestCase):
    def test_same_literal_on_all_edges_folds_phi_to_const(self):
        fn_ = _fold(_merge_program(2, 2)).functions[0]
        self.assertFalse(any(b.phis for b in fn_.blocks))
        merge = fn_.blocks[3]
        # The phi became a Const of the same SSA value at the top of the
        # merge block's ordinary instructions.
        first = merge.instructions[0]
        self.assertIsInstance(first, Const)
        self.assertEqual(first.value, 2)
        self.assertIs(merge.terminator.value, first.dest)

    def test_distinct_literals_keep_phi(self):
        fn_ = _fold(_merge_program(2, 3)).functions[0]
        merge = fn_.blocks[3]
        self.assertEqual(len(merge.phis), 1)
        phi = merge.phis[0]
        self.assertIs(merge.terminator.value, phi.dest)
        self.assertEqual(len(phi.entries), 2)

    def test_unknown_input_keeps_phi(self):
        fn_ = _fold(program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", int_(0)),
                if_(var("c"),
                    [assign("a", int_(2))],
                    [assign("a", var("x"))]),
                ret(var("a")),
            ],
        ))).functions[0]
        merge = fn_.blocks[3]
        self.assertEqual(len(merge.phis), 1)

    def test_loop_carried_phi_is_not_folded(self):
        fn_ = _fold(program(func(
            "f", [param("n", "int")], "int",
            [
                let("i", "int", int_(0)),
                let("acc", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")), [
                    assign("acc", arith("add", var("acc"), var("i"))),
                    assign("i", arith("add", var("i"), int_(1))),
                ]),
                ret(var("acc")),
            ],
        ))).functions[0]
        header = fn_.blocks[1]
        self.assertEqual(len(header.phis), 2)
        self.assertTrue(
            any(isinstance(i, BinOp) for i in _instructions(fn_))
        )

    def test_phi_folded_through_another_phi(self):
        # Both branches assign the same folded expression; the merge phi
        # sees two distinct but equally-valued Const definitions.
        fn_ = _fold(program(func(
            "f", [param("c", "bool")], "int",
            [
                let("a", "int", int_(0)),
                if_(var("c"),
                    [assign("a", arith("add", int_(1), int_(1)))],
                    [assign("a", arith("mul", int_(1), int_(2)))]),
                ret(var("a")),
            ],
        ))).functions[0]
        self.assertFalse(any(b.phis for b in fn_.blocks))
        merge = fn_.blocks[3]
        self.assertIsInstance(merge.instructions[0], Const)
        self.assertEqual(merge.instructions[0].value, 2)


# --------------------------------------------------------------------------
# Structural preservation
# --------------------------------------------------------------------------


class PreservationTests(unittest.TestCase):
    def test_function_block_and_terminator_layout_preserved(self):
        ast = program(
            func("consts", [param("c", "bool")], "int",
                 [ret(arith("add", int_(1), int_(2)))]),
            _merge_program(2, 3)["functions"][0],
            func("g", [], "bool", [ret(bool_(True))]),
        )
        ssa = to_ssa(lower_module(ast))
        result = fold_constants(ssa)
        self.assertEqual([f.name for f in result.functions],
                         [f.name for f in ssa.functions])
        for old, new in zip(ssa.functions, result.functions):
            self.assertEqual(new.ret_type, old.ret_type)
            self.assertEqual([p.name for p in new.params],
                             [p.name for p in old.params])
            self.assertEqual([b.id for b in new.blocks],
                             [b.id for b in old.blocks])
            for old_b, new_b in zip(old.blocks, new.blocks):
                self.assertEqual(type(new_b.terminator),
                                 type(old_b.terminator))

    def test_ssa_numbers_are_preserved(self):
        ssa = to_ssa(lower_module(_merge_program(2, 2)))
        before = {d.id for d in _all_defs(ssa.functions[0])}
        result = fold_constants(ssa)
        after = {d.id for d in _all_defs(result.functions[0])}
        self.assertEqual(before, after)

    def test_folded_phi_keeps_its_ssa_number(self):
        ssa = to_ssa(lower_module(_merge_program(2, 2)))
        phi = ssa.functions[0].blocks[3].phis[0]
        result = fold_constants(ssa)
        const = result.functions[0].blocks[3].instructions[0]
        self.assertIsInstance(const, Const)
        self.assertEqual(const.dest.id, phi.dest.id)

    def test_unfolded_instruction_relative_order_preserved(self):
        ssa = to_ssa(lower_module(_echo_program([
            let("a", "int", call("echo", [int_(1)])),
            let("b", "int", call("echo", [arith("add", var("a"), int_(2))])),
            ret(arith("add", var("a"), var("b"))),
        ])))
        result = fold_constants(ssa).functions[0]
        old_unfolded = [
            type(i).__name__
            for b in ssa.functions[0].blocks for i in b.instructions
            if isinstance(i, Call)
            or (isinstance(i, BinOp))
        ]
        new_kinds = [
            type(i).__name__ for i in _instructions(result)
            if isinstance(i, (Call, BinOp))
        ]
        self.assertEqual(old_unfolded, new_kinds)

    def test_constant_branch_is_not_rewritten(self):
        fn_ = _fold(program(func(
            "f", [], "int",
            [if_(bool_(True), [ret(int_(1))], [ret(int_(2))])],
        ))).functions[0]
        self.assertIsInstance(fn_.blocks[0].terminator, Branch)
        # No block is deleted either.
        self.assertEqual(len(fn_.blocks), 3)

    def test_output_is_ssa_closed(self):
        for ast in (
            _foldable_ast(),
            _merge_program(2, 2),
            _merge_program(2, 3),
            _echo_program([ret(call("echo", [int_(5)]))]),
        ):
            assert_ssa_closed(self, _fold(ast))


# --------------------------------------------------------------------------
# Fixpoint, idempotence and determinism
# --------------------------------------------------------------------------


def _rich_program():
    return program(
        func("rich", [param("c", "bool"), param("x", "int")], "int", [
            let("k", "int", arith("mul", arith("add", int_(1), int_(2)),
                                  int_(3))),
            let("a", "int", int_(0)),
            if_(var("c"),
                [assign("a", arith("add", var("k"), var("k")))],
                [assign("a", arith("sub", var("k"), int_(9)))]),
            let("v", "int", call("echo", [var("a")])),
            ret(arith("add", var("v"), arith("div", var("k"), int_(2)))),
        ]),
        func("echo", [param("x", "int")], "int", [ret(var("x"))]),
    )


class IdempotenceTests(unittest.TestCase):
    PROGRAMS = [
        _foldable_ast(),
        _merge_program(2, 2),
        _merge_program(2, 3),
        _rich_program(),
        _echo_program([ret(call("echo", [int_(5)]))]),
    ]

    def test_repeated_folding_is_stable(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            once = fold_constants(ssa)
            twice = fold_constants(once)
            self.assertEqual(render_module(once), render_module(twice))
            self.assertTrue(
                _container_objects(once).isdisjoint(
                    _container_objects(twice))
            )
            thrice = fold_constants(twice)
            self.assertEqual(render_module(twice), render_module(thrice))

    def test_render_is_deterministic(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            self.assertEqual(
                render_module(fold_constants(ssa)),
                render_module(fold_constants(ssa)),
            )


# --------------------------------------------------------------------------
# Observable semantics: values, call traces and runtime faults preserved
# --------------------------------------------------------------------------


class SemanticPreservationTests(unittest.TestCase):
    def test_existing_cases_keep_return_and_call_trace(self):
        for name, ast, entry, runs in _SEMANTIC_CASES:
            for arguments, _expected_return, _expected_trace in runs:
                with self.subTest(case=name, arguments=arguments):
                    module = lower_module(ast)
                    expected = _interpret_plain(module, entry, arguments)
                    folded = fold_constants(to_ssa(module))
                    self.assertEqual(
                        _interpret_plain(folded, entry, arguments),
                        expected,
                    )

    def test_fault_cases_keep_category_site_and_operands(self):
        for label, builder, entry, arguments, pinned in _ORDER_CASES:
            with self.subTest(sample=label):
                folded = fold_constants(to_ssa(lower_module(builder())))
                outcome = _interpret_outcome(folded, entry, arguments)
                self.assertEqual(outcome, pinned)
                if pinned.kind == "fault":
                    self.assertEqual(outcome.category, ZERO_DIVISOR)

    def test_fold_then_dead_code_elimination_is_safe(self):
        for label, builder, entry, arguments, pinned in _ORDER_CASES:
            with self.subTest(sample=label):
                module = lower_module(builder())
                cleaned = eliminate_dead_code(
                    fold_constants(to_ssa(module))
                )
                assert_ssa_closed(self, cleaned)
                self.assertEqual(
                    _interpret_outcome(cleaned, entry, arguments), pinned
                )

    def test_folded_constants_are_cleaned_up_by_dce(self):
        # The folded-away 1+2 chain and the unused k*2 leave dead Consts
        # behind; DCE removes them without touching the observable core.
        cleaned = eliminate_dead_code(fold_constants(to_ssa(lower_module(
            program(func(
                "f", [], "int",
                [
                    let("d", "int", arith("add", int_(1), int_(2))),
                    ret(int_(5)),
                ],
            ))
        ))))
        fn_ = cleaned.functions[0]
        self.assertEqual(
            [(i.value,) for i in _instructions(fn_)
             if isinstance(i, Const)],
            [(5,)],
        )
        self.assertFalse(any(isinstance(i, BinOp) for i in _instructions(fn_)))


if __name__ == "__main__":
    unittest.main()
