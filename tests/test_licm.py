"""Tests for the SSA loop-invariant code motion entry point.

These pin :func:`hoist_loop_invariants` on six contracts:

1. API -- public export; non-:class:`Module` raises ``TypeError``;
   non-SSA :class:`Module` raises ``ValueError``; the empty module and
   loop-free modules come back as independent, content-equivalent copies.
2. Hoisting rules -- ``Const`` and add/sub/mul/comparison BinOps whose
   operands are loop-invariant move to the unique preheader, in the
   original global instruction order and before the preheader
   terminator; ``Phi``, ``Call``, ``Copy`` and div/mod never move.
3. Shape gate -- a header without a unique outside predecessor, or whose
   outside predecessor branches elsewhere, keeps its loop untouched; no
   new block is ever created.
4. Nesting -- inner loops are handled before outer loops, an invariant
   of both lands in the outermost preheader in one move, and a
   gate-failing contained loop is a barrier to its enclosing loops.
5. Determinism / idempotence / independence -- repeated calls and the
   new default pipeline reach a structural and textual fixed point,
   share no mutable container with the input, and never mutate it.
6. Semantics -- return values, ordered call traces/arguments and the
   site and pre-fault trace of div/mod traps are preserved, including
   zero-trip loops.
"""
import copy
import unittest

from compiler_ir import (
    BinOp,
    Block,
    Branch,
    Call,
    Const,
    Function,
    Jump,
    Module,
    Parameter,
    Return,
    Slot,
    Temp,
    hoist_loop_invariants,
    lower_module,
    optimize_module,
    render_module,
    to_ssa,
)

from test_pipeline import (
    arith,
    assign,
    call,
    compare,
    func,
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
    _CASES,
    _clean_program,
    _interpret,
    _loop_program,
)
from test_dce import _container_objects, assert_ssa_closed


def _ssa(builder):
    return to_ssa(lower_module(copy.deepcopy(builder())))


def _function(module, name):
    return next(f for f in module.functions if f.name == name)


def _blocks(fn):
    return {b.label: b for b in fn.blocks}


class ApiContractTests(unittest.TestCase):
    def test_public_export(self):
        import compiler_ir

        self.assertIn("hoist_loop_invariants", compiler_ir.__all__)
        self.assertIs(
            compiler_ir.hoist_loop_invariants, hoist_loop_invariants)

    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    hoist_loop_invariants(bad)

    def test_non_ssa_module_raises_value_error(self):
        lowered = lower_module(copy.deepcopy(_clean_program()))
        self.assertFalse(lowered.ssa)
        with self.assertRaises(ValueError):
            hoist_loop_invariants(lowered)
        with self.assertRaises(ValueError):
            hoist_loop_invariants(Module([], ssa=False))

    def test_empty_module_returns_independent_copy(self):
        empty = to_ssa(lower_module(program()))
        result = hoist_loop_invariants(empty)
        self.assertIsNot(result, empty)
        self.assertTrue(result.ssa)
        self.assertEqual(result.functions, [])
        self.assertEqual(
            render_module(result), render_module(empty))

    def test_loop_free_module_returns_equivalent_disjoint_copy(self):
        ssa = _ssa(_clean_program)
        result = hoist_loop_invariants(ssa)
        self.assertIsNot(result, ssa)
        self.assertEqual(render_module(result), render_module(ssa))
        self.assertTrue(
            _container_objects(ssa).isdisjoint(_container_objects(result)))

    def test_input_is_never_mutated(self):
        ssa = _ssa(_loop_program)
        before = render_module(ssa)
        hoist_loop_invariants(ssa)
        self.assertEqual(render_module(ssa), before)

    def test_result_shares_no_mutable_container(self):
        for builder in _loop_builders():
            ssa = _ssa(builder)
            result = hoist_loop_invariants(ssa)
            with self.subTest(builder=builder.__name__):
                self.assertTrue(
                    _container_objects(ssa).isdisjoint(
                        _container_objects(result)))
                assert_ssa_closed(self, result)


def _loop_builders():
    return [_loop_program] + [
        case[1] for case in _CASES
        if "loop" in case[0]
    ]


class HoistingRulesTests(unittest.TestCase):
    def test_invariant_const_and_arith_move_to_preheader(self):
        # base = k + 1 and its constant 1 are loop invariant; the body's
        # i + 1 step is not.
        ssa = _ssa(_loop_program)
        result = hoist_loop_invariants(ssa)
        caller = _function(result, "loopsum")
        blocks = _blocks(caller)

        # b0 is the single preheader of the header b1.
        self.assertEqual(
            [i.operator for i in blocks["b0"].instructions
             if isinstance(i, BinOp)],
            ["add"],
        )
        # The invariant add's operands are the parameter k and a Const 1,
        # both also present in the preheader.
        invariant_add = next(
            i for i in blocks["b0"].instructions
            if isinstance(i, BinOp) and i.operator == "add")
        owners = {
            id(i.dest): i for b in caller.blocks
            for i in b.instructions
        }
        self.assertIsInstance(owners[id(invariant_add.right)], Const)
        self.assertEqual(owners[id(invariant_add.right)].value, 1)
        # Every preheader instruction precedes its terminator.
        self.assertIsNotNone(blocks["b0"].terminator)

    def test_hoisted_order_follows_original_global_order(self):
        # Two independent invariants 7 and k+7 chain; the const precedes
        # the add in the body lowering, so it precedes it in the
        # preheader too.
        ast = program(
            func("ord", [param("n", "int"), param("k", "int")], "int", [
                let("s", "int", int_(0)),
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")), [
                    let("a", "int", arith("add", int_(7), var("k"))),
                    assign("s", arith("add", var("s"), var("a"))),
                    assign("i", arith("add", var("i"), int_(1))),
                ]),
                ret(var("s")),
            ]),
        )
        ssa = to_ssa(lower_module(ast))
        result = hoist_loop_invariants(ssa)
        pre = _blocks(_function(result, "ord"))["b0"]
        const7 = next(i for i in pre.instructions
                      if isinstance(i, Const) and i.value == 7)
        invariant_add = next(
            i for i in pre.instructions
            if isinstance(i, BinOp) and i.operator == "add"
            and i.left is const7.dest)
        self.assertLess(
            pre.instructions.index(const7),
            pre.instructions.index(invariant_add))
        # All hoisted instructions sit ahead of the (unchanged) jump to
        # the header; the body's later invariant const 1 follows the
        # 7 + k chain in the original global order.
        self.assertIsNotNone(pre.terminator)

    def test_phi_call_and_div_mod_never_move(self):
        ast = program(
            func("keep", [param("n", "int"), param("x", "int")], "int", [
                let("s", "int", int_(0)),
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")), [
                    # invariant comparison
                    let("cmp", "bool", compare("lt", var("x"), int_(9))),
                    # invariant call must stay (zero-trip safety)
                    let("e", "int", call("ping", [var("x")])),
                    # invariant div must stay (trap safety)
                    let("q", "int", arith("div", var("x"), int_(2))),
                    # invariant mod must stay
                    let("r", "int", arith("mod", var("x"), int_(3))),
                    assign("s", arith(
                        "add",
                        arith("add", arith("add", var("s"), var("e")),
                              var("q")),
                        arith("add", var("r"),
                              arith("add", var("i"),
                                    int_(0))))),
                    assign("i", arith("add", var("i"), int_(1))),
                ]),
                ret(var("s")),
            ]),
            func("ping", [param("v", "int")], "int", [ret(var("v"))]),
        )
        ssa = to_ssa(lower_module(ast))
        result = hoist_loop_invariants(ssa)
        caller = _function(result, "keep")
        header_blocks = {"b1", "b2"}

        def loop_body_instructions():
            for label in header_blocks:
                block = _blocks(caller).get(label)
                if block is not None:
                    for i in block.instructions:
                        yield i

        ops = [(type(i).__name__,
                getattr(i, "operator", None)) for i in
               loop_body_instructions()]
        # Call and div/mod remain somewhere in the loop region.
        self.assertIn(("Call", None), ops)
        self.assertIn(("BinOp", "div"), ops)
        self.assertIn(("BinOp", "mod"), ops)
        # The header phis survive.
        self.assertTrue(
            any(b.phis for b in caller.blocks if b.label in header_blocks))

    def test_zero_trip_loop_gains_no_call(self):
        ast = program(
            func("z", [param("n", "int")], "int", [
                let("s", "int", int_(0)),
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")), [
                    let("e", "int", arith("add", int_(2), int_(3))),
                    let("c", "int", call("ping", [var("e")])),
                    assign("s", arith("add", var("s"), var("c"))),
                    assign("i", arith("add", var("i"), int_(1))),
                ]),
                ret(var("s")),
            ]),
            func("ping", [param("v", "int")], "int", [ret(var("v"))]),
        )
        ssa = to_ssa(lower_module(ast))
        result = hoist_loop_invariants(ssa)
        self.assertEqual(
            _interpret(ssa, "z", (0,)),
            _interpret(result, "z", (0,)),
        )
        outcome = _interpret(result, "z", (0,))
        self.assertEqual(outcome.output, [])
        self.assertEqual(outcome.value, 0)

    def test_no_new_blocks_created(self):
        ssa = _ssa(_loop_program)
        result = hoist_loop_invariants(ssa)
        for before, after in zip(ssa.functions, result.functions):
            self.assertEqual(
                [b.id for b in before.blocks],
                [b.id for b in after.blocks])


class ShapeGateTests(unittest.TestCase):
    @staticmethod
    def _module_with_two_outside_preds():
        p0, side, header, latch, exit_ = [Block(i) for i in range(5)]
        cond = Temp(0, "bool")
        k = Temp(1, "int")
        p0.terminator = Branch(cond, header, side)
        side.terminator = Jump(header)
        header.instructions.append(Const(k, 5))
        header.terminator = Branch(cond, latch, exit_)
        latch.terminator = Jump(header)
        exit_.terminator = Return(None)
        fn = Function(
            "f", [Parameter("c", Slot(0, "bool"), cond)], "void", [],
            [p0, side, header, latch, exit_], p0, ssa=True)
        return Module([fn], ssa=True)

    def test_two_outside_predecessors_blocks_hoisting(self):
        module = self._module_with_two_outside_preds()
        result = hoist_loop_invariants(module)
        header = next(b for b in result.functions[0].blocks if b.id == 2)
        self.assertTrue(
            any(isinstance(i, Const) for i in header.instructions),
            "the invariant must stay when the gate fails")
        self.assertEqual(
            render_module(result),
            render_module(hoist_loop_invariants(result)))

    def test_preheader_branching_elsewhere_blocks_hoisting(self):
        p0, header, latch, exit_, side = [Block(i) for i in range(5)]
        cond = Temp(0, "bool")
        k = Temp(1, "int")
        # The sole outside predecessor branches to the header and to a
        # side block, so code moved "before the header" could run without
        # entering the loop in a context where the side path converges
        # back; the gate rejects this shape outright.
        side.terminator = Jump(exit_)
        p0.terminator = Branch(cond, header, side)
        header.instructions.append(Const(k, 5))
        header.terminator = Branch(cond, latch, exit_)
        latch.terminator = Jump(header)
        exit_.terminator = Return(None)
        fn = Function(
            "g", [Parameter("c", Slot(0, "bool"), cond)], "void", [],
            [p0, header, latch, exit_, side], p0, ssa=True)
        module = Module([fn], ssa=True)
        result = hoist_loop_invariants(module)
        header = next(b for b in result.functions[0].blocks if b.id == 1)
        self.assertTrue(
            any(isinstance(i, Const) for i in header.instructions))


class NestedLoopTests(unittest.TestCase):
    @staticmethod
    def _nested_ast():
        return program(
            func("nest",
                 [param("n", "int"), param("m", "int"),
                  param("k", "int")], "int", [
                let("s", "int", int_(0)),
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")), [
                    let("j", "int", int_(0)),
                    while_(compare("lt", var("j"), var("m")), [
                        let("a", "int", arith("add", int_(7), var("k"))),
                        assign("s", arith("add", var("s"), var("a"))),
                        assign("j", arith("add", var("j"), int_(1))),
                    ]),
                    assign("i", arith("add", var("i"), int_(1))),
                ]),
                ret(var("s")),
            ]),
        )

    def test_invariant_of_both_loops_reaches_outer_preheader(self):
        ssa = to_ssa(lower_module(self._nested_ast()))
        result = hoist_loop_invariants(ssa)
        caller = _function(result, "nest")
        blocks = _blocks(caller)
        # The 7 and 7 + k chain are invariant to the inner and the outer
        # loop and land in the outermost preheader b0 in one move.
        self.assertIn(
            7, [i.value for i in blocks["b0"].instructions
                if isinstance(i, Const)])
        self.assertTrue(
            any(isinstance(i, BinOp) and i.operator == "add"
                for i in blocks["b0"].instructions))
        inner_header = blocks["b4"]
        inner_body = blocks["b5"]
        # The Const 7 no longer exists anywhere inside either loop.
        inside = [b for b in caller.blocks
                  if b.label in {"b1", "b2", "b4", "b5", "b6"}]
        self.assertFalse(
            any(isinstance(i, Const) and i.value == 7
                for b in inside for i in b.instructions))
        # The inner body keeps exactly its two variant adds (s + a and
        # j + 1); the invariant 7 + k add is gone.
        body_adds = [i for i in inner_body.instructions
                     if isinstance(i, BinOp) and i.operator == "add"]
        self.assertEqual(len(body_adds), 2)
        const7 = next(i for i in blocks["b0"].instructions
                      if isinstance(i, Const) and i.value == 7)
        for add in body_adds:
            self.assertIsNot(add.right, const7.dest)
            self.assertIsNot(add.left, const7.dest)

    def test_each_instruction_moves_once_and_is_a_fixed_point(self):
        ssa = to_ssa(lower_module(self._nested_ast()))
        once = hoist_loop_invariants(ssa)
        twice = hoist_loop_invariants(once)
        self.assertEqual(render_module(once), render_module(twice))

    def test_nested_semantics(self):
        ssa = to_ssa(lower_module(self._nested_ast()))
        result = hoist_loop_invariants(ssa)
        for args in ((0, 0, 3), (2, 3, 10), (1, 5, -2), (3, 0, 4)):
            self.assertEqual(
                _interpret(ssa, "nest", args),
                _interpret(result, "nest", args))


class DeterminismAndSemanticsTests(unittest.TestCase):
    def test_byte_identical_across_runs(self):
        for builder in _loop_builders():
            ssa = _ssa(builder)
            texts = [
                render_module(hoist_loop_invariants(ssa)).encode()
                for _ in range(3)
            ]
            self.assertTrue(texts[0] == texts[1] == texts[2])

    def test_observables_match_baseline(self):
        for label, builder, entry, args, _pin in _CASES:
            ssa = _ssa(builder)
            result = hoist_loop_invariants(ssa)
            with self.subTest(sample=label):
                self.assertEqual(
                    _interpret(ssa, entry, args),
                    _interpret(result, entry, args))

    def test_default_pipeline_is_fixed_point(self):
        for builder in _loop_builders():
            lowered = lower_module(copy.deepcopy(builder()))
            once = optimize_module(lowered)
            twice = optimize_module(once)
            with self.subTest(builder=builder.__name__):
                self.assertEqual(
                    render_module(once), render_module(twice))
                self.assertEqual(DEFAULT_ORDER,
                                 ("ssa", "fold", "licm", "dce", "ssa"))

    def test_licm_repeated_in_pipeline_reaches_fixed_point(self):
        lowered = lower_module(copy.deepcopy(_loop_program()))
        once = optimize_module(
            lowered, ("ssa", "fold", "licm", "dce", "ssa"))
        twice = optimize_module(
            lowered, ("ssa", "fold", "licm", "licm", "dce", "ssa"))
        self.assertEqual(render_module(once), render_module(twice))

    def test_licm_stage_prevalidated_in_pipeline(self):
        lowered = lower_module(copy.deepcopy(_loop_program()))
        text = render_module(lowered)
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("licm",))
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("fold", "licm"))
        # A late licm after a bogus name still fails before any pass runs.
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("ssa", "bogus", "licm"))
        self.assertEqual(render_module(lowered), text)

    def test_unknown_name_and_stage_fail_before_execution(self):
        lowered = lower_module(copy.deepcopy(_loop_program()))
        for order in (("ssa", "licm", "nope"), ("licm", "dce")):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    optimize_module(lowered, order)

    def test_invariant_before_faulting_div_keeps_fault_site(self):
        # An invariant pure add textually precedes a faulting div in the
        # same loop block.  It must NOT move, or the div's within-block
        # ordinal (the numbering-independent fault site) would shift.
        ast = program(
            func("lf", [param("n", "int"), param("x", "int")], "int", [
                let("s", "int", int_(0)),
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")), [
                    let("a", "int", arith("add", var("x"), int_(1))),
                    let("e", "int", call("emit", [var("i")])),
                    let("q", "int", arith(
                        "div", int_(100),
                        arith("sub", int_(2), var("i")))),
                    assign("s", arith(
                        "add", var("s"),
                        arith("add", var("q"), var("a")))),
                    assign("i", arith("add", var("i"), int_(1))),
                ]),
                ret(var("s")),
            ]),
            func("emit", [param("v", "int")], "int", [ret(var("v"))]),
        )
        ssa = to_ssa(lower_module(copy.deepcopy(ast)))
        result = hoist_loop_invariants(ssa)
        body = _blocks(_function(result, "lf"))["b2"]
        # The invariant add stays in the body, ahead of the faulting div.
        body_ops = [
            (type(i).__name__, getattr(i, "operator", None))
            for i in body.instructions]
        add_index = body_ops.index(("BinOp", "add"))
        div_index = body_ops.index(("BinOp", "div"))
        self.assertLess(add_index, div_index)
        # Fault site, operands and pre-fault trace are unchanged.
        self.assertEqual(
            _interpret(ssa, "lf", (5, 7)),
            _interpret(result, "lf", (5, 7)))
        outcome = _interpret(result, "lf", (5, 7))
        self.assertEqual(outcome.site, ("lf", "b2", "div", 2))
        self.assertEqual(
            outcome.output,
            [("emit", (0,)), ("emit", (1,)), ("emit", (2,))])
        # The call is unaffected, and a non-faulting run still agrees.
        self.assertEqual(
            _interpret(ssa, "lf", (2, 7)),
            _interpret(result, "lf", (2, 7)))
        # The fence must not disturb the fixed point.
        self.assertEqual(
            render_module(result),
            render_module(hoist_loop_invariants(result)))

    def test_fold_then_licm_exposes_folded_invariant(self):
        # The folded loop-invariant Const 6 is what licm then hoists.
        from test_pass_ordering import _loop_step_program

        lowered = lower_module(copy.deepcopy(_loop_step_program()))
        result = optimize_module(
            lowered, ("ssa", "fold", "licm", "dce", "ssa"))
        caller = _function(result, "loopstep")
        pre_consts = [
            i.value for i in _blocks(caller)["b0"].instructions
            if isinstance(i, Const)]
        self.assertIn(6, pre_consts)
        for args in ((0,), (5,), (20,)):
            self.assertEqual(
                _interpret(to_ssa(lowered), "loopstep", args),
                _interpret(result, "loopstep", args))


if __name__ == "__main__":
    unittest.main()
