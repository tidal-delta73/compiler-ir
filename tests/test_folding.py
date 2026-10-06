"""Tests for SSA constant folding and propagation (:func:`fold_constants`).

The pass is tested seven ways:

* the API contract -- new SSA module, input never mutated, ``TypeError``
  for non-modules, ``ValueError`` for non-SSA modules, independent copies
  for the empty module and for modules with nothing foldable, and a
  top-level export;
* literal folding -- constant arithmetic chains (all five operators,
  truncation-toward-zero ``div``/``mod``), constant comparisons (all six,
  yielding bool), and propagation of folded results into later defs;
* non-folding rules -- unknown operands keep the ``BinOp``, a known zero
  divisor keeps the faulting ``BinOp`` (the trap is neither advanced,
  swallowed nor rewritten), and ``Call`` results are always unknown;
* phi handling -- same-literal merges fold (even when the literals are
  produced by folded arithmetic), any disagreement or unknown edge keeps
  the phi, and a loop header phi with a constant entry edge but a
  disagreeing back edge is correctly left unfolded (lattice fixpoint);
* preservation -- function/block/terminator/phi/instruction order, SSA
  numbers, constant branches left as branches, no blocks removed, and the
  output stays SSA-closed;
* fixpoint, idempotence and determinism -- repeated folding is a
  structural and textual fixed point and independent runs are
  byte-identical, and the pass composes safely with
  :func:`eliminate_dead_code`;
* semantics -- the shared IR interpreters execute the folded module (and
  the folded-then-DCE module) and observe the same return value, call
  sequence and runtime fault as the unoptimized input.
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
from test_dce import _container_objects, assert_ssa_closed
from test_pass_ordering import _CASES as _ORDER_CASES
from test_pass_ordering import _interpret as _interpret_outcome
from test_semantic_equivalence import _CASES as _SEM_CASES
from test_semantic_equivalence import _interpret as _interpret_trace


# --------------------------------------------------------------------------
# Structural helpers
# --------------------------------------------------------------------------


def _all_defs(fn_):
    defs = [p.temp for p in fn_.params]
    for block in fn_.blocks:
        defs.extend(phi.dest for phi in block.phis)
        defs.extend(ins.dest for ins in block.instructions)
    return defs


def _const_values(fn_):
    return [ins.value for block in fn_.blocks for ins in block.instructions
            if isinstance(ins, Const)]


def _fold(ast):
    return fold_constants(to_ssa(lower_module(ast)))


# --------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------


def _foldable_program():
    return program(func(
        "f", [param("x", "int")], "int",
        [
            let("a", "int", arith("add", int_(1), int_(2))),
            let("b", "int", arith("mul", var("a"), int_(4))),
            ret(arith("add", var("b"), var("x"))),
        ],
    ))


class ApiTests(unittest.TestCase):
    def test_returns_new_module_marked_ssa(self):
        ssa = to_ssa(lower_module(_foldable_program()))
        result = fold_constants(ssa)
        self.assertIsInstance(result, Module)
        self.assertIsNot(result, ssa)
        self.assertTrue(result.ssa)
        self.assertTrue(all(f.ssa for f in result.functions))

    def test_exported_from_package_top_level(self):
        import compiler_ir

        self.assertIs(compiler_ir.fold_constants, fold_constants)
        self.assertIn("fold_constants", compiler_ir.__all__)

    def test_input_module_is_not_mutated(self):
        ssa = to_ssa(lower_module(_foldable_program()))
        before = render_module(ssa)
        fold_constants(ssa)
        self.assertEqual(render_module(ssa), before)

    def test_shares_no_mutable_containers_with_input(self):
        ssa = to_ssa(lower_module(_foldable_program()))
        result = fold_constants(ssa)
        self.assertTrue(
            _container_objects(ssa).isdisjoint(_container_objects(result))
        )

    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.assertRaises(TypeError):
                fold_constants(bad)

    def test_non_ssa_module_raises_value_error(self):
        non_ssa = lower_module(_foldable_program())
        self.assertFalse(non_ssa.ssa)
        with self.assertRaises(ValueError):
            fold_constants(non_ssa)
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
        self.assertTrue(
            _container_objects(empty).isdisjoint(_container_objects(result))
        )

    def test_unfoldable_module_returns_independent_equivalent_copy(self):
        # A loop counter: only plain Consts and unknown-operand BinOps, no
        # foldable BinOp and no same-literal merge phi.
        ast = program(func(
            "f", [param("n", "int")], "int",
            [
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")),
                       [assign("i", arith("add", var("i"), int_(1)))]),
                ret(var("i")),
            ],
        ))
        ssa = to_ssa(lower_module(ast))
        once = fold_constants(ssa)
        self.assertEqual(render_module(once), render_module(ssa))
        self.assertIsNot(once, ssa)
        self.assertTrue(
            _container_objects(ssa).isdisjoint(_container_objects(once))
        )


# --------------------------------------------------------------------------
# Arithmetic and comparison folding
# --------------------------------------------------------------------------


_ARITH_EXPECT = {
    "add": (-7 + 2, 7 + -2),
    "sub": (-7 - 2, 7 - -2),
    "mul": (-7 * 2, 7 * -2),
    "div": (-3, -3),   # truncation toward zero
    "mod": (-1, 1),    # a == div(a,b)*b + mod(a,b)
}


class ArithmeticFoldingTests(unittest.TestCase):
    def test_every_arith_operator_folds_with_truncation_semantics(self):
        for operator, (neg_left, neg_right) in _ARITH_EXPECT.items():
            with self.subTest(operator=operator):
                ast = program(func(
                    "f", [], "int",
                    [
                        let("a", "int", arith(operator, int_(-7), int_(2))),
                        let("b", "int", arith(operator, int_(7), int_(-2))),
                        ret(arith("add", var("a"), var("b"))),
                    ],
                ))
                fn_ = _fold(ast).functions[0]
                consts = _const_values(fn_)
                self.assertIn(neg_left, consts)
                self.assertIn(neg_right, consts)
                self.assertIn(neg_left + neg_right, consts)
                # No arithmetic BinOp remains at all.
                self.assertFalse(any(
                    isinstance(i, BinOp) and i.kind == "arith"
                    for b in fn_.blocks for i in b.instructions
                ))

    def test_div_mod_identity_over_mixed_signs(self):
        # Folded values must satisfy a == div(a,b)*b + mod(a,b) exactly.
        for a in (-17, -7, -1, 0, 1, 7, 17):
            for b in (-5, -2, 2, 5):
                ast = program(func(
                    "f", [], "int",
                    [
                        let("q", "int", arith("div", int_(a), int_(b))),
                        let("r", "int", arith("mod", int_(a), int_(b))),
                        ret(arith(
                            "sub",
                            int_(a),
                            arith("add", arith("mul", var("q"), int_(b)),
                                  var("r")),
                        )),
                    ],
                ))
                fn_ = _fold(ast).functions[0]
                returned = fn_.blocks[0].terminator.value
                owner = next(i for i in fn_.blocks[0].instructions
                             if i.dest.id == returned.id)
                self.assertIsInstance(owner, Const)
                self.assertEqual(owner.value, 0)

    def test_unknown_operand_keeps_binop_but_known_side_still_folds(self):
        ast = program(func(
            "f", [param("x", "int")], "int",
            [
                let("k", "int", int_(10)),
                let("y", "int", arith("add", var("x"), var("k"))),
                ret(arith("mul", var("y"), int_(3))),
            ],
        ))
        fn_ = _fold(ast).functions[0]
        block = fn_.blocks[0]
        add = next(i for i in block.instructions
                   if isinstance(i, BinOp) and i.operator == "add")
        mul = next(i for i in block.instructions
                   if isinstance(i, BinOp) and i.operator == "mul")
        # Both BinOps survive: the add depends on parameter x, and the mul
        # depends on the add.
        self.assertEqual(add.left.type, "int")
        # The mul's right operand names the folded constant 3's slot.
        right = next(i for i in block.instructions
                     if i.dest is mul.right)
        self.assertIsInstance(right, Const)
        self.assertEqual(right.value, 3)
        self.assertTrue(any(i.value == 10 for i in block.instructions
                            if isinstance(i, Const)))

    def test_folded_result_propagates_into_later_definitions(self):
        # (6 * 7) folds first; the subtraction using it then folds too.
        ast = program(func(
            "f", [], "int",
            [ret(arith("sub", arith("mul", int_(6), int_(7)), int_(2)))],
        ))
        fn_ = _fold(ast).functions[0]
        ops = [i for b in fn_.blocks for i in b.instructions
               if isinstance(i, BinOp)]
        self.assertEqual(ops, [])
        self.assertEqual(fn_.blocks[0].terminator.value.type, "int")
        final = next(i for i in fn_.blocks[0].instructions
                     if i.dest is fn_.blocks[0].terminator.value)
        self.assertEqual(final.value, 40)


class ComparisonFoldingTests(unittest.TestCase):
    def test_all_comparisons_fold_to_bool(self):
        cases = {
            "eq": (3, 3, True), "ne": (3, 3, False),
            "lt": (2, 3, True), "le": (3, 3, True),
            "gt": (4, 3, True), "ge": (2, 3, False),
        }
        for operator, (a, b, expected) in cases.items():
            with self.subTest(operator=operator):
                ast = program(func(
                    "f", [], "bool",
                    [ret(compare(operator, int_(a), int_(b)))],
                ))
                fn_ = _fold(ast).functions[0]
                self.assertFalse(any(
                    isinstance(i, BinOp) for i in fn_.blocks[0].instructions
                ))
                last = fn_.blocks[0].instructions[-1]
                self.assertIsInstance(last, Const)
                self.assertEqual(last.dest.type, "bool")
                self.assertEqual(last.value, expected)
                self.assertEqual(
                    fn_.blocks[0].terminator.value.type, "bool"
                )

    def test_folded_comparison_feeds_branch_without_becoming_a_jump(self):
        # The condition is a compile-time true constant; folding propagates
        # it, but the Branch terminator must remain a Branch.
        ast = program(func(
            "f", [param("x", "int")], "int",
            [
                if_(compare("lt", int_(1), int_(2)),
                    [ret(int_(10))],
                    [ret(int_(20))]),
            ],
        ))
        result = _fold(ast)
        fn_ = result.functions[0]
        entry = fn_.entry
        self.assertIsInstance(entry.terminator, Branch)
        # The condition slot is a bool Const definition.
        cond = entry.terminator.cond
        owner = next(i for b in fn_.blocks for i in b.instructions
                     if i.dest is cond)
        self.assertIsInstance(owner, Const)
        self.assertEqual(owner.value, True)
        self.assertEqual(owner.dest.type, "bool")


# --------------------------------------------------------------------------
# Zero divisors and calls
# --------------------------------------------------------------------------


class ZeroDivisorTests(unittest.TestCase):
    def _folded(self, operator):
        ast = program(func(
            "f", [], "int",
            [
                let("a", "int", arith("mul", int_(6), int_(7))),
                let("z", "int", int_(0)),
                let("q", "int", arith(operator, var("a"), var("z"))),
                ret(arith("sub", var("q"), int_(5))),
            ],
        ))
        return _fold(ast).functions[0]

    def test_known_zero_divisor_binop_is_never_folded(self):
        for operator in ("div", "mod"):
            with self.subTest(operator=operator):
                fn_ = self._folded(operator)
                faulting = [
                    i for b in fn_.blocks for i in b.instructions
                    if isinstance(i, BinOp) and i.operator == operator
                ]
                self.assertEqual(len(faulting), 1)
                ins = faulting[0]
                # Its operands were folded (42 and 0) but the trap itself
                # stayed a BinOp in its original slot.
                left = next(x for b in fn_.blocks for x in b.instructions
                            if getattr(x, "dest", None) is ins.left)
                right = next(x for b in fn_.blocks for x in b.instructions
                             if getattr(x, "dest", None) is ins.right)
                self.assertEqual(left.value, 42)
                self.assertEqual(right.value, 0)
                # The dependent subtraction cannot fold either.
                self.assertTrue(any(
                    isinstance(i, BinOp) and i.operator == "sub"
                    for b in fn_.blocks for i in b.instructions
                ))

    def test_fault_still_fires_at_same_block_with_same_operands(self):
        for operator in ("div", "mod"):
            ast = program(func(
                "f", [], "int",
                [
                    let("a", "int", arith("mul", int_(6), int_(7))),
                    let("z", "int", int_(0)),
                    ret(arith(operator, var("a"), var("z"))),
                ],
            ))
            lowered = lower_module(ast)
            folded = fold_constants(to_ssa(lowered))
            baseline = _interpret_outcome(lowered, "f", ())
            actual = _interpret_outcome(folded, "f", ())
            self.assertEqual(actual.kind, "fault")
            self.assertEqual(baseline.kind, "fault")
            self.assertEqual(actual.category, baseline.category)
            self.assertEqual(actual.operands, (42, 0))
            self.assertEqual(actual.output, baseline.output)
            # The trap is on the entry block and keeps its operator.
            self.assertEqual(actual.site[0], "f")
            self.assertEqual(actual.site[1], "b0")
            self.assertEqual(actual.site[2], operator)


class CallTests(unittest.TestCase):
    @staticmethod
    def _id_function():
        return func("id", [param("v", "int")], "int", [ret(var("v"))])

    def test_call_result_is_unknown_and_order_is_kept(self):
        ast = program(
            func("run", [], "int", [
                let("r", "int",
                    arith("add", call("id", [int_(4)]), int_(1))),
                ret(var("r")),
            ]),
            self._id_function(),
        )
        fn_ = _fold(ast).functions[0]
        calls = [i for b in fn_.blocks for i in b.instructions
                 if isinstance(i, Call)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "id")
        # The add consuming the call cannot fold...
        adds = [i for b in fn_.blocks for i in b.instructions
                if isinstance(i, BinOp) and i.operator == "add"]
        self.assertEqual(len(adds), 1)
        # ...but the call's constant argument propagated (still a Const).
        self.assertEqual(calls[0].args[0].type, "int")
        arg_owner = next(
            i for b in fn_.blocks for i in b.instructions
            if i.dest is calls[0].args[0]
        )
        self.assertIsInstance(arg_owner, Const)
        self.assertEqual(arg_owner.value, 4)
        # The call precedes the add in the block instruction stream.
        block = fn_.blocks[0]
        kinds = [type(i).__name__ for i in block.instructions]
        self.assertLess(kinds.index("Call"), kinds.index("BinOp"))

    def test_unused_call_is_retained(self):
        ast = program(
            func("run", [param("x", "int")], "void", [
                let("ignored", "int", call("id", [var("x")])),
                let("d", "int", arith("add", int_(1), int_(2))),
            ]),
            self._id_function(),
        )
        fn_ = _fold(ast).functions[0]
        self.assertEqual(
            [i.name for b in fn_.blocks for i in b.instructions
             if isinstance(i, Call)],
            ["id"],
        )


# --------------------------------------------------------------------------
# Phi folding
# --------------------------------------------------------------------------


class PhiFoldingTests(unittest.TestCase):
    def test_same_literal_phi_folds_even_via_folded_arithmetic(self):
        # Both arms compute the literal 7 (3+4 and 10-3); the merge phi
        # folds to Const 7 and the following multiply folds to 14.
        ast = program(func(
            "f", [param("c", "bool")], "int",
            [
                let("a", "int", int_(7)),
                if_(var("c"),
                    [assign("a", arith("add", int_(3), int_(4)))],
                    [assign("a", arith("sub", int_(10), int_(3)))]),
                ret(arith("mul", var("a"), int_(2))),
            ],
        ))
        fn_ = _fold(ast).functions[0]
        self.assertFalse(any(phi for b in fn_.blocks for phi in b.phis))
        # The merge block is the one holding the folded phi Const (7) and
        # the folded return multiply (14).
        merge = next(
            b for b in fn_.blocks
            if {i.value for i in b.instructions if isinstance(i, Const)}
            >= {7, 14}
        )
        # The folded phi Const comes first in its block, in phi position.
        first = merge.instructions[0]
        self.assertIsInstance(first, Const)
        self.assertEqual(first.value, 7)

    def test_differing_literals_keep_the_phi(self):
        ast = program(func(
            "f", [param("c", "bool")], "int",
            [
                let("a", "int", int_(1)),
                if_(var("c"), [assign("a", int_(2))], []),
                ret(var("a")),
            ],
        ))
        fn_ = _fold(ast).functions[0]
        phis = [phi for b in fn_.blocks for phi in b.phis]
        self.assertEqual(len(phis), 1)
        self.assertEqual(len(phis[0].entries), 2)

    def test_one_unknown_edge_keeps_the_phi(self):
        # One arm assigns a, the other leaves the parameter-derived value.
        ast = program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", var("x")),
                if_(var("c"), [assign("a", int_(5))], []),
                ret(arith("mul", var("a"), int_(2))),
            ],
        ))
        fn_ = _fold(ast).functions[0]
        phis = [phi for b in fn_.blocks for phi in b.phis]
        self.assertEqual(len(phis), 1)
        # The downstream multiply cannot fold while the merge is unknown.
        self.assertTrue(any(
            isinstance(i, BinOp) and i.operator == "mul"
            for b in fn_.blocks for i in b.instructions
        ))

    def test_loop_header_phi_constant_entry_but_backedge_disagrees(self):
        # i starts at 0; the body advances it by the folded 1+5=6.  The
        # header phi looks constant on the entry edge but the back edge
        # disagrees, so it must survive.
        ast = program(func(
            "f", [param("n", "int")], "int",
            [
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")),
                       [assign("i", arith(
                           "add", var("i"), arith("add", int_(1), int_(5))))]),
                ret(var("i")),
            ],
        ))
        fn_ = _fold(ast).functions[0]
        header = fn_.blocks[1]
        self.assertEqual(len(header.phis), 1)
        phi = header.phis[0]
        self.assertEqual(len(phi.entries), 2)
        # The loop-invariant 1+5 folded; the i+6 add stays (i unknown).
        body = fn_.blocks[2]
        consts = {i.value for i in body.instructions if isinstance(i, Const)}
        self.assertIn(6, consts)
        adds = [i for i in body.instructions
                if isinstance(i, BinOp) and i.operator == "add"]
        self.assertEqual(len(adds), 1)
        # Semantics with step 6: f(20) walks 0,6,12,18,24 -> returns 24.
        folded = fold_constants(to_ssa(lower_module(ast)))
        self.assertEqual(_interpret_outcome(folded, "f", (20,)).value, 24)
        self.assertEqual(_interpret_outcome(folded, "f", (0,)).value, 0)

    def test_surviving_phi_keeps_entries_and_predecessor_order(self):
        ast = program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", int_(1)),
                if_(var("c"),
                    [assign("a", arith("add", var("x"), int_(2)))],
                    [assign("a", arith("mul", var("x"), int_(3)))]),
                ret(var("a")),
            ],
        ))
        ssa = to_ssa(lower_module(ast))
        fn_ = fold_constants(ssa).functions[0]
        phi = next(phi for b in fn_.blocks for phi in b.phis)
        keys = list(phi.entries.keys())
        self.assertEqual([k.id for k in keys], sorted(k.id for k in keys))
        self.assertEqual([k.label for k in keys], ["b1", "b2"])


# --------------------------------------------------------------------------
# Structural preservation
# --------------------------------------------------------------------------


class PreservationTests(unittest.TestCase):
    PROGRAMS = [_foldable_program()] + [
        builder() for _label, builder, _e, _a, _p in _ORDER_CASES
    ]

    def test_function_order_signature_and_block_layout_preserved(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            result = fold_constants(ssa)
            with self.subTest(program=ast):
                self.assertEqual(
                    [f.name for f in result.functions],
                    [f.name for f in ssa.functions],
                )
                for old, new in zip(ssa.functions, result.functions):
                    self.assertEqual(new.ret_type, old.ret_type)
                    self.assertEqual(
                        [(p.name, p.slot) for p in new.params],
                        [(p.name, p.slot) for p in old.params],
                    )
                    self.assertEqual(new.entry.id, old.entry.id)
                    self.assertEqual(
                        [b.id for b in new.blocks],
                        [b.id for b in old.blocks],
                    )
                    for old_b, new_b in zip(old.blocks, new.blocks):
                        self.assertEqual(
                            type(new_b.terminator),
                            type(old_b.terminator),
                        )

    def test_all_ssa_numbers_are_preserved(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            result = fold_constants(ssa)
            with self.subTest(program=ast):
                for old_f, new_f in zip(ssa.functions, result.functions):
                    self.assertEqual(
                        sorted(d.id for d in _all_defs(old_f)),
                        sorted(d.id for d in _all_defs(new_f)),
                    )

    def test_ordinary_instructions_keep_dest_order_and_slot_kinds(self):
        ast = program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", arith("add", int_(3), int_(4))),
                if_(var("c"),
                    [assign("a", int_(7))],
                    [assign("a", arith("sub", int_(10), int_(3)))]),
                ret(arith("mul", var("a"), int_(2))),
            ],
        ))
        ssa = to_ssa(lower_module(ast))
        result = fold_constants(ssa)
        for old_b, new_b in zip(ssa.functions[0].blocks,
                                result.functions[0].blocks):
            # Every original ordinary instruction is still there, in the
            # same dest order (possibly rewritten BinOp -> Const), after
            # any leading Consts introduced for folded phis.
            folded_phi_count = len(old_b.phis) - len(new_b.phis)
            self.assertGreaterEqual(
                len(new_b.instructions) - len(old_b.instructions), 0,
            )
            self.assertEqual(
                len(new_b.instructions) - len(old_b.instructions),
                folded_phi_count,
            )
            tail = new_b.instructions[folded_phi_count:]
            self.assertEqual(
                [i.dest.id for i in tail],
                [i.dest.id for i in old_b.instructions],
            )
            for old_i, new_i in zip(old_b.instructions, tail):
                if isinstance(old_i, BinOp) and isinstance(new_i, Const):
                    continue  # an actual fold
                self.assertEqual(type(new_i), type(old_i))

    def test_constant_branch_is_not_rewritten_to_jump(self):
        ast = program(func(
            "f", [], "int",
            [if_(compare("lt", int_(1), int_(2)),
                 [ret(int_(1))], [ret(int_(2))])],
        ))
        result = _fold(ast).functions[0]
        self.assertIsInstance(result.entry.terminator, Branch)
        # No block disappeared either.
        self.assertEqual(
            [b.id for b in result.blocks],
            [b.id for b in to_ssa(lower_module(ast)).functions[0].blocks],
        )

    def test_output_is_ssa_closed(self):
        for ast in self.PROGRAMS:
            assert_ssa_closed(self, fold_constants(to_ssa(lower_module(ast))))


# --------------------------------------------------------------------------
# Fixpoint, idempotence, determinism and composition with DCE
# --------------------------------------------------------------------------


class IdempotenceTests(unittest.TestCase):
    PROGRAMS = [_foldable_program()] + [
        builder() for _label, builder, _e, _a, _p in _ORDER_CASES
    ]

    def test_repeated_folding_is_stable(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            once = fold_constants(ssa)
            twice = fold_constants(once)
            with self.subTest(program=ast):
                self.assertEqual(
                    render_module(once), render_module(twice)
                )
                for f1, f2 in zip(once.functions, twice.functions):
                    self.assertEqual(
                        sorted(d.id for d in _all_defs(f1)),
                        sorted(d.id for d in _all_defs(f2)),
                    )
                thrice = fold_constants(twice)
                self.assertEqual(
                    render_module(twice), render_module(thrice)
                )
                self.assertIsNot(twice, once)

    def test_render_is_deterministic(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            texts = [render_module(fold_constants(ssa)) for _ in range(3)]
            self.assertTrue(all(t == texts[0] for t in texts))
            self.assertTrue(
                all(t.encode("utf-8") == texts[0].encode("utf-8")
                    for t in texts)
            )

    def test_composes_with_dce_and_keeps_fixed_point(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            folded = fold_constants(ssa)
            cleaned = eliminate_dead_code(folded)
            with self.subTest(program=ast):
                # DCE removes folded-then-unused definitions; folding the
                # cleaned module changes nothing further.
                self.assertEqual(
                    render_module(fold_constants(cleaned)),
                    render_module(cleaned),
                )
                # fold after the canonical ssa renumbering is fixed too.
                canonical = to_ssa(cleaned)
                self.assertEqual(
                    render_module(fold_constants(canonical)),
                    render_module(canonical),
                )

    def test_fold_dce_then_interpret_matches_baseline(self):
        for label, builder, entry, args, _pinned in _ORDER_CASES:
            ast = builder()
            lowered = lower_module(ast)
            baseline = _interpret_outcome(lowered, entry, args)
            optimized = eliminate_dead_code(
                fold_constants(to_ssa(lowered))
            )
            with self.subTest(case=label, args=args):
                self.assertEqual(
                    _interpret_outcome(optimized, entry, args), baseline
                )
                assert_ssa_closed(self, optimized)


# --------------------------------------------------------------------------
# Observable semantics across the existing case libraries
# --------------------------------------------------------------------------


class SemanticPreservationTests(unittest.TestCase):
    def test_ordering_cases_keep_outcome_including_faults(self):
        for label, builder, entry, args, pinned in _ORDER_CASES:
            ast = builder()
            lowered = lower_module(ast)
            folded = fold_constants(to_ssa(lowered))
            with self.subTest(case=label, args=args):
                baseline = _interpret_outcome(lowered, entry, args)
                actual = _interpret_outcome(folded, entry, args)
                self.assertEqual(actual, baseline)
                # The pin anchors the fault sites/traces independently.
                self.assertEqual(actual, pinned)
                assert_ssa_closed(self, folded)

    def test_semantic_equivalence_cases_keep_return_and_call_trace(self):
        for name, ast, entry, runs in _SEM_CASES:
            for args, expected_return, expected_trace in runs:
                lowered = lower_module(ast)
                folded = fold_constants(to_ssa(lowered))
                with self.subTest(case=name, args=args):
                    value, trace = _interpret_trace(folded, entry, args)
                    self.assertEqual(value, expected_return)
                    self.assertEqual(trace, expected_trace)


if __name__ == "__main__":
    unittest.main()
