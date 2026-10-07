"""Tests for loop-invariant code motion (:func:`hoist_loop_invariants`).

The pass is tested six ways:

* the API contract -- a brand new SSA module, input never mutated,
  ``TypeError`` for non-modules, ``ValueError`` for non-SSA modules, and
  independent content-equivalent copies for the empty module and for
  modules without a hoistable loop;
* natural-loop discovery and the shape gate -- a header with one
  loop-external predecessor that flows only to the header; headers with
  two external predecessors, a branching sole predecessor, no external
  predecessor, or an unreachable back edge keep every instruction in
  place;
* hoistable kinds -- ``Const`` always; ``add``/``sub``/``mul`` and
  comparisons once operands are outside the loop or already hoisted;
  ``Phi``, ``Call``, ``Copy``, ``div`` and ``mod`` never move;
* placement and stability -- instructions land before the preheader's
  terminator in original block/instruction order, nested loops are
  handled inner to outer in one pass, each instruction physically moves
  once, and SSA numbers/labels/terminators/function order survive;
* determinism / idempotence -- repeated calls are byte-identical and the
  pass is a structural and textual fixed point, also when driven through
  ``optimize_module``'s default schedule;
* semantics -- zero-trip loops gain no call and no early division/modulo
  fault, while return values, ordered call traces and fault sites match
  the unoptimized baseline.
"""
import copy
import unittest

from compiler_ir import (
    BinOp,
    Block,
    Branch,
    Call,
    Const,
    Copy,
    Function,
    Jump,
    Module,
    Parameter,
    Phi,
    Return,
    Slot,
    Temp,
    hoist_loop_invariants,
    lower_module,
    optimize_module,
    render_module,
    to_ssa,
)

from test_dce import _container_objects
from test_pass_ordering import (
    _interpret,
    _loop_fault_program,
    _loop_step_program,
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


# ==========================================================================
# Hand-built SSA construction helpers
# ==========================================================================


def _temp(number: int, typ="int") -> Temp:
    return Temp(number, typ)


def _param(name: str, temp: Temp) -> Parameter:
    """A parameter sharing the exact Temp object the body references."""
    return Parameter(name, Slot(temp.id, temp.type), temp)


def _function(blocks, entry_index=0, params=()):
    entry = blocks[entry_index]
    return Function(
        "f", list(params), "int", [], list(blocks), entry, ssa=True
    )


def _kinds(block):
    return [type(ins).__name__ for ins in block.instructions]


def _ids(block):
    return [ins.dest.id for ins in block.instructions]


# ==========================================================================
# API contract
# ==========================================================================


class HoistApiContractTests(unittest.TestCase):
    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    hoist_loop_invariants(bad)

    def test_non_ssa_module_raises_value_error(self):
        lowered = lower_module(_loop_step_program())
        self.assertFalse(lowered.ssa)
        with self.assertRaises(ValueError):
            hoist_loop_invariants(lowered)

    def test_empty_module_is_independent_equivalent_copy(self):
        empty = Module([], ssa=True)
        result = hoist_loop_invariants(empty)
        self.assertIsNot(result, empty)
        self.assertTrue(result.ssa)
        self.assertEqual(result.functions, [])
        self.assertEqual(render_module(result), render_module(empty))
        self.assertTrue(
            _container_objects(empty).isdisjoint(
                _container_objects(result)))

    def test_loop_free_module_is_independent_equivalent_copy(self):
        ssa = to_ssa(lower_module(program(func(
            "f", [param("x", "int")], "int",
            [let("y", "int", arith("add", var("x"), int_(1))),
             ret(var("y"))]))))
        before = render_module(ssa)
        result = hoist_loop_invariants(ssa)
        self.assertEqual(render_module(result), before)
        self.assertIsNot(result, ssa)
        self.assertTrue(
            _container_objects(ssa).isdisjoint(
                _container_objects(result)))

    def test_input_is_never_mutated_and_shares_no_container(self):
        ssa = to_ssa(lower_module(_loop_step_program()))
        before = render_module(ssa)
        result = hoist_loop_invariants(ssa)
        self.assertEqual(render_module(ssa), before)
        self.assertTrue(
            _container_objects(ssa).isdisjoint(
                _container_objects(result)))
        # Mutating the result cannot reach the input.
        result.functions[0].blocks[0].instructions.append(
            result.functions[0].blocks[0].instructions[0])
        self.assertEqual(render_module(ssa), before)

    def test_function_order_is_preserved(self):
        ssa = to_ssa(lower_module(program(
            func("second", [param("x", "int")], "int",
                 [let("y", "int", arith("add", var("x"), int_(1))),
                  ret(var("y"))]),
            func("first_loop", [param("n", "int")], "int", [
                let("i", "int", int_(0)),
                while_(compare("lt", var("i"), var("n")),
                       [assign("i", arith("add", var("i"), int_(1)))]),
                ret(var("i")),
            ]),
        )))
        result = hoist_loop_invariants(ssa)
        self.assertEqual(
            [f.name for f in result.functions], ["second", "first_loop"])


# ==========================================================================
# Shape gate
# ==========================================================================


class HoistShapeGateTests(unittest.TestCase):
    def _assert_unchanged(self, module):
        result = hoist_loop_invariants(module)
        self.assertEqual(render_module(result), render_module(module))
        self.assertTrue(
            _container_objects(module).isdisjoint(
                _container_objects(result)))

    def test_header_with_two_external_predecessors_is_left_intact(self):
        # b0 branches to the header b1 and to b4, which also jumps to b1;
        # the header additionally has the internal back edge b2 -> b1.
        b0, b1, b2, b3, b4 = (Block(i) for i in range(5))
        p0 = Temp(0, "bool")
        t1, t2 = Temp(1, "int"), Temp(2, "bool")
        v3, v4, v5 = Temp(3, "int"), Temp(4, "int"), Temp(5, "int")
        b0.instructions.append(Const(t1, 0))
        b0.terminator = Branch(p0, b1, b4)
        b1.phis.append(Phi(t1, {b0: v3, b4: v4, b2: v5}))
        # A loop-invariant const that must NOT move.
        b1.instructions.append(Const(t2, True))
        b1.terminator = Branch(t2, b2, b3)
        b2.instructions.append(BinOp(v5, "add", t1, t1, "arith", "int"))
        b2.terminator = Jump(b1)
        b3.terminator = Return(t1)
        b4.instructions.append(Const(v4, 1))
        b4.terminator = Jump(b1)
        b0.instructions.insert(0, Const(v3, 0))
        module = Module([_function(
            [b0, b1, b2, b3, b4],
            params=(_param("c", p0),))], ssa=True)
        self._assert_unchanged(module)

    def test_branching_preheader_is_left_intact(self):
        # The header b1 has one external predecessor b0, but b0 also
        # branches straight to the exit b3, so nothing may move.
        b0, b1, b2, b3 = (Block(i) for i in range(4))
        p0 = Temp(0, "bool")
        cond = Temp(1, "bool")
        value = Temp(2, "int")
        back = Temp(3, "int")
        b0.terminator = Branch(p0, b1, b3)
        b1.instructions.append(Const(cond, True))
        b1.instructions.append(Const(value, 42))  # invariant, must stay
        b1.terminator = Branch(cond, b2, b3)
        b2.instructions.append(BinOp(back, "add", value, value,
                                     "arith", "int"))
        b2.terminator = Jump(b1)
        b3.terminator = Return(value)
        module = Module([_function(
            [b0, b1, b2, b3],
            params=(_param("c", p0),))], ssa=True)
        self._assert_unchanged(module)

    def test_entry_block_loop_header_is_left_intact(self):
        # The entry doubles as the loop header: it has no external
        # predecessor at all.
        b0, b1, b2 = Block(0), Block(1), Block(2)
        p0 = Temp(0, "bool")
        cond = Temp(1, "bool")
        value = Temp(2, "int")
        b0.instructions.append(Const(cond, True))
        b0.instructions.append(Const(value, 7))  # must stay
        b0.terminator = Branch(cond, b1, b2)
        b1.terminator = Jump(b0)
        b2.terminator = Return(value)
        module = Module([_function(
            [b0, b1, b2],
            params=(_param("c", p0),))], ssa=True)
        self._assert_unchanged(module)

    def test_unreachable_loop_is_left_intact(self):
        # b0 returns immediately; b1/b2 form an unreachable loop.
        b0, b1, b2 = Block(0), Block(1), Block(2)
        cond = Temp(0, "bool")
        value = Temp(1, "int")
        b0.instructions.append(Const(cond, True))
        b0.terminator = Return(cond)
        b1.instructions.append(Const(value, 9))
        b1.terminator = Branch(cond, b2, b0)
        b2.terminator = Jump(b1)
        module = Module([_function(
            [b0, b1, b2],
            params=(_param("c", cond),))], ssa=True)
        self._assert_unchanged(module)
        result = hoist_loop_invariants(module)
        # The invariant const stays in the unreachable header.
        self.assertEqual(
            _ids(result.functions[0].blocks[1]), [value.id])


# ==========================================================================
# Hoistable vs. pinned instructions
# ==========================================================================


def _gated_loop_module():
    """A canonical while-shaped loop exercising every instruction kind.

    b0 is the unique preheader (jumps only to header b1).  b2 is the loop
    body; b3 the exit.  Body instructions (see ids) mix hoistable chains,
    loop-variant operands, faulting arithmetic and a call.
    """
    b0, b1, b2, b3 = Block(0), Block(1), Block(2), Block(3)
    p0 = Temp(0, "int")
    t1, t2, t3 = Temp(1, "int"), Temp(2, "int"), Temp(3, "int")
    iv = Temp(4, "int")
    cond = Temp(5, "bool")
    back = Temp(6, "int")
    # Hoistable candidates in the body.
    c7 = Temp(7, "int")
    c8 = Temp(8, "int")
    c9 = Temp(9, "int")
    c10 = Temp(10, "bool")
    # Pinned instructions.
    v11 = Temp(11, "int")   # add using the header phi
    v12 = Temp(12, "int")   # div with loop-outside operands
    v13 = Temp(13, "int")   # mod with loop-outside operands
    v14 = Temp(14, "int")   # call
    c15 = Temp(15, "int")   # copy

    b0.instructions.extend([Const(t1, 10), Const(t2, 20)])
    b0.instructions.append(BinOp(t3, "add", t1, t2, "arith", "int"))
    b0.terminator = Jump(b1)

    b1.phis.append(Phi(iv, {b0: t1, b2: back}))
    b1.instructions.append(
        BinOp(cond, "lt", iv, p0, "compare", "bool"))
    b1.terminator = Branch(cond, b2, b3)

    b2.instructions.extend([
        Const(c7, 7),
        BinOp(c8, "add", c7, t1, "arith", "int"),
        BinOp(c9, "mul", t3, c8, "arith", "int"),
        BinOp(c10, "eq", c7, t2, "compare", "bool"),
        BinOp(v11, "sub", iv, c7, "arith", "int"),
        BinOp(v12, "div", c8, t2, "arith", "int"),
        BinOp(v13, "mod", t1, t2, "arith", "int"),
        Call(v14, "emit", [c8], "int"),
        Copy(c15, c7),
        BinOp(back, "add", iv, t1, "arith", "int"),
    ])
    b2.terminator = Jump(b1)
    b3.terminator = Return(iv)

    module = Module([_function(
        [b0, b1, b2, b3],
        params=(_param("n", p0),))], ssa=True)
    return module, (c7, c8, c9, c10), (v11, v12, v13, v14, c15, back)


class HoistKindsTests(unittest.TestCase):
    def test_const_and_safe_binop_chains_hoist_others_stay(self):
        module, hoisted, pinned = _gated_loop_module()
        result = hoist_loop_invariants(module)
        fn_ = result.functions[0]
        b0, b1, b2, b3 = fn_.blocks

        # The four safe invariants move as a chain into the preheader,
        # after its original instructions and before the jump.
        self.assertEqual(
            [ins.dest.id for ins in b0.instructions],
            [1, 2, 3, 7, 8, 9, 10])
        kinds = _kinds(b0)
        self.assertEqual(kinds[-4:], ["Const", "BinOp", "BinOp", "BinOp"])
        self.assertIsInstance(b0.terminator, Jump)
        self.assertIs(b0.terminator.target, b1)

        # Body keeps the phi-dependent sub, the faulting div/mod, the call
        # and the copy; the backedge add stays.  Relative order preserved.
        self.assertEqual(
            [ins.dest.id for ins in b2.instructions],
            [11, 12, 13, 14, 15, 6])
        self.assertEqual(
            _kinds(b2),
            ["BinOp", "BinOp", "BinOp", "Call", "Copy", "BinOp"])

        # The header and its phi/branch are untouched.
        self.assertEqual(len(b1.phis), 1)
        self.assertEqual([ins.dest.id for ins in b1.instructions], [5])
        self.assertIsInstance(b3.terminator, Return)

    def test_each_hoisted_instruction_keeps_its_ssa_number(self):
        module, hoisted, _pinned = _gated_loop_module()
        result = hoist_loop_invariants(module)
        b0 = result.functions[0].blocks[0]
        self.assertEqual(
            [ins.dest.id for ins in b0.instructions[-4:]],
            [temp.id for temp in hoisted])

    def test_hoisted_instruction_physically_moves_once(self):
        module, _h, _p = _gated_loop_module()
        result = hoist_loop_invariants(module)
        blocks = result.functions[0].blocks
        for temp_id in (7, 8, 9, 10):
            owners = [
                b.label for b in blocks
                if any(ins.dest.id == temp_id for ins in b.instructions)]
            self.assertEqual(owners, ["b0"])

    def test_comparison_with_variant_operand_stays(self):
        # The header condition compares the loop phi: it must not move.
        module, _h, _p = _gated_loop_module()
        result = hoist_loop_invariants(module)
        b1 = result.functions[0].blocks[1]
        self.assertEqual([ins.dest.id for ins in b1.instructions], [5])

    def test_multiple_back_edges_share_one_loop_and_preheader(self):
        b0, b1, b2, b3, b4 = (Block(i) for i in range(5))
        t1, t2 = Temp(1, "bool"), Temp(2, "bool")
        c3, c4 = Temp(3, "int"), Temp(4, "int")
        back = Temp(5, "int")
        seed = Temp(6, "int")
        b0.instructions.append(Const(seed, 0))
        b0.terminator = Jump(b1)
        b1.phis.append(Phi(back, {b0: seed, b2: c3, b3: c4}))
        b1.instructions.append(Const(t1, True))
        b1.instructions.append(Const(t2, True))
        b1.terminator = Branch(t1, b2, b4)
        b2.instructions.append(Const(c3, 3))
        b2.terminator = Branch(t2, b1, b3)
        b3.instructions.append(Const(c4, 4))
        b3.terminator = Jump(b1)
        b4.terminator = Return(back)
        module = Module([_function([b0, b1, b2, b3, b4])], ssa=True)
        result = hoist_loop_invariants(module)
        blocks = result.functions[0].blocks
        # All four loop-invariant consts hoist, in global origin order
        # (header b1 first, then latches b2 and b3), after b0's original
        # seed const.
        self.assertEqual(
            [ins.dest.id for ins in blocks[0].instructions],
            [6, 1, 2, 3, 4])
        self.assertEqual(_ids(blocks[1]), [])
        self.assertEqual(_ids(blocks[2]), [])
        self.assertEqual(_ids(blocks[3]), [])


# ==========================================================================
# Nested loops
# ==========================================================================


def _nested_loop_module():
    # b0 -> b1 (outer header) -> b6 (inner preheader) -> b2 (inner header)
    # b2 -> b3 (inner body) -> b2 ; b2 -> b4 (latch) -> b1
    # b1 -> b5 outer exit.
    blocks = [Block(i) for i in range(7)]
    b0, b1, b2, b3, b4, b5, b6 = blocks
    p0 = Temp(0, "int")
    seed = Temp(1, "int")
    outer_iv = Temp(2, "int")
    outer_cond = Temp(3, "bool")
    inner_seed = Temp(4, "int")
    inner_iv = Temp(5, "int")
    inner_cond = Temp(6, "bool")
    outer_latch = Temp(7, "int")
    # b3 instructions:
    t10, t11, t12, t13 = (Temp(i, "int") for i in range(10, 14))

    b0.instructions.append(Const(seed, 0))
    b0.instructions.append(Const(inner_seed, 0))
    b0.terminator = Jump(b1)

    b1.phis.append(Phi(outer_iv, {b0: seed, b4: outer_latch}))
    b1.instructions.append(
        BinOp(outer_cond, "lt", outer_iv, p0, "compare", "bool"))
    b1.terminator = Branch(outer_cond, b6, b5)

    b6.terminator = Jump(b2)

    b2.phis.append(Phi(inner_iv, {b6: inner_seed, b3: t13}))
    b2.instructions.append(
        BinOp(inner_cond, "lt", inner_iv, p0, "compare", "bool"))
    b2.terminator = Branch(inner_cond, b3, b4)

    b3.instructions.extend([
        Const(t10, 3),                          # invariant to both
        BinOp(t11, "add", t10, p0, "arith", "int"),   # chain: both
        BinOp(t12, "add", outer_iv, t10, "arith", "int"),  # inner only
        BinOp(t13, "add", inner_iv, t10, "arith", "int"),  # variant
    ])
    b3.terminator = Jump(b2)

    b4.instructions.append(
        BinOp(outer_latch, "add", outer_iv, inner_iv, "arith", "int"))
    b4.terminator = Jump(b1)

    b5.terminator = Return(outer_iv)

    fn_ = _function(blocks, params=(_param("n", p0),))
    module = Module([fn_], ssa=True)
    return module, blocks


class NestedLoopTests(unittest.TestCase):
    def test_inner_to_outer_hoisting_in_one_pass(self):
        module, blocks = _nested_loop_module()
        result = hoist_loop_invariants(module)
        out = result.functions[0].blocks
        b0, b1, b2, b3, b4, b5, b6 = out

        # Invariant to both loops lands straight in the outer preheader,
        # keeping body instruction order (const 10 then add 11).
        self.assertEqual([ins.dest.id for ins in b0.instructions],
                         [1, 4, 10, 11])
        # The value invariant to the inner loop but variant to the outer
        # (uses the outer header phi) stops at the inner preheader b6.
        self.assertEqual([ins.dest.id for ins in b6.instructions], [12])
        # Only the truly inner-variant add remains in the inner body.
        self.assertEqual([ins.dest.id for ins in b3.instructions], [13])
        # Headers and the outer latch are otherwise unchanged.
        self.assertEqual(len(b1.phis), 1)
        self.assertEqual(len(b2.phis), 1)
        self.assertEqual([ins.dest.id for ins in b4.instructions], [7])
        self.assertIsInstance(b5.terminator, Return)

    def test_nested_hoist_is_a_one_pass_fixed_point(self):
        module, _blocks = _nested_loop_module()
        once = hoist_loop_invariants(module)
        twice = hoist_loop_invariants(once)
        self.assertEqual(render_module(once), render_module(twice))
        self.assertEqual(
            [ins.dest.id for ins in twice.functions[0].blocks[0].instructions],
            [1, 4, 10, 11])


# ==========================================================================
# Determinism and fixpoints
# ==========================================================================


class HoistDeterminismTests(unittest.TestCase):
    def test_repeated_calls_are_byte_identical(self):
        ssa = to_ssa(lower_module(_loop_step_program()))
        texts = [
            render_module(hoist_loop_invariants(ssa)).encode("utf-8")
            for _ in range(3)
        ]
        self.assertEqual(texts[0], texts[1])
        self.assertEqual(texts[1], texts[2])

    def test_pass_is_structural_and_textual_fixed_point(self):
        for builder in (_loop_step_program, _loop_fault_program):
            ssa = to_ssa(lower_module(builder()))
            once = hoist_loop_invariants(ssa)
            twice = hoist_loop_invariants(once)
            with self.subTest(builder=builder.__name__):
                self.assertEqual(
                    render_module(once), render_module(twice))
                self.assertIsNot(twice, once)
                self.assertTrue(
                    _container_objects(once).isdisjoint(
                        _container_objects(twice)))

    def test_folded_constant_hoists_and_keeps_numbering(self):
        # fold turns the invariant `1 + 5` into Const 6 inside the body;
        # licm (without a trailing renumbering) moves that very slot out.
        from compiler_ir import fold_constants

        folded = fold_constants(
            to_ssa(lower_module(_loop_step_program())))
        hoisted = hoist_loop_invariants(folded)
        fn_ = next(
            f for f in hoisted.functions if f.name == "loopstep")
        preheader, header, body, exit_ = fn_.blocks
        pre_consts = [ins.value for ins in preheader.instructions
                      if isinstance(ins, Const)]
        body_consts = [ins.value for ins in body.instructions
                       if isinstance(ins, Const)]
        self.assertIn(6, pre_consts)
        self.assertNotIn(6, body_consts)
        # The surviving i + 6 add still references the hoisted slot.
        [add] = [ins for ins in body.instructions
                 if isinstance(ins, BinOp)]
        self.assertEqual(add.right.id,
                         next(ins.dest.id for ins in preheader.instructions
                              if isinstance(ins, Const) and ins.value == 6))


# ==========================================================================
# Pipeline integration
# ==========================================================================


class HoistPipelineTests(unittest.TestCase):
    def test_licm_is_scheduled_in_the_default_order(self):
        from compiler_ir.pipeline import DEFAULT_PASSES

        self.assertEqual(
            DEFAULT_PASSES, ("ssa", "fold", "licm", "dce", "ssa"))

    def test_default_pipeline_is_a_fixed_point(self):
        lowered = lower_module(_loop_step_program())
        once = optimize_module(lowered)
        twice = optimize_module(once)
        self.assertEqual(render_module(once), render_module(twice))

    def test_repeated_licm_in_explicit_schedule_is_stable(self):
        lowered = lower_module(_loop_step_program())
        once = optimize_module(
            lowered, ("ssa", "fold", "licm", "dce", "ssa"))
        twice = optimize_module(
            lowered,
            ("ssa", "fold", "licm", "licm", "dce", "ssa"))
        self.assertEqual(render_module(once), render_module(twice))

    def test_licm_before_ssa_rejected_before_any_pass_runs(self):
        lowered = lower_module(_loop_step_program())
        text = render_module(lowered)
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("licm",))
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("ssa", "fold", "bogus", "licm"))
        # Validation is whole-order and up front.
        self.assertEqual(render_module(lowered), text)

    def test_unknown_name_remains_value_error(self):
        ssa = to_ssa(lower_module(_loop_step_program()))
        with self.assertRaises(ValueError):
            optimize_module(ssa, ("licm", "inline"))


# ==========================================================================
# Semantics: zero trips, calls, and division/modulo faults
# ==========================================================================


def _invariant_call_program():
    # A pure folded invariant (7 * 8 -> 56) feeds a call that must remain
    # inside the loop; the zero-trip input must observe no call.
    return program(
        func("main", [param("n", "int")], "int", [
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("a", "int", arith("mul", int_(7), int_(8))),
                let("e", "int", call("emit", [var("a")])),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            ret(int_(0)),
        ]),
        func("emit", [param("v", "int")], "int", [ret(var("v"))]),
    )


def _invariant_fault_program(operator):
    # The divisor folds to literal 0 and is itself loop invariant; the
    # faulting BinOp must stay in the live loop body (its result feeds the
    # loop-carried total returned after exit) so the zero-trip path returns
    # normally and a taken iteration faults only after the preceding call.
    return program(
        func("main", [param("n", "int"), param("x", "int")], "int", [
            let("total", "int", int_(0)),
            let("i", "int", int_(0)),
            while_(compare("lt", var("i"), var("n")), [
                let("e", "int", call("emit", [var("i")])),
                let("q", "int",
                    arith(operator, var("x"), arith("sub", int_(1), int_(1)))),
                assign("total", arith("add", var("total"), var("q"))),
                assign("i", arith("add", var("i"), int_(1))),
            ]),
            ret(var("total")),
        ]),
        func("emit", [param("v", "int")], "int", [ret(var("v"))]),
    )


class HoistSemanticsTests(unittest.TestCase):
    def test_zero_trip_loop_gains_no_call(self):
        lowered = lower_module(_invariant_call_program())
        result = optimize_module(lowered)
        # Structurally the folded const 56 is pre-loop and the call is not.
        main = next(f for f in result.functions if f.name == "main")
        rendered = render_module(result)
        self.assertIn("56", rendered)
        body = next(
            b for b in main.blocks
            if any(isinstance(i, Call) for i in b.instructions))
        body_consts = [
            i.value for i in body.instructions if isinstance(i, Const)]
        self.assertNotIn(56, body_consts)
        # Zero trips: nothing observed.
        self.assertEqual(
            _interpret(result, "main", (0,)).output, [])
        # Taken trips: the call still executes each iteration in order.
        self.assertEqual(
            _interpret(result, "main", (3,)).output,
            [("emit", (56,)), ("emit", (56,)), ("emit", (56,))])

    def test_invariant_zero_divisor_does_not_fault_on_zero_trip(self):
        for operator in ("div", "mod"):
            lowered = lower_module(_invariant_fault_program(operator))
            result = optimize_module(lowered)
            with self.subTest(operator=operator):
                outcome = _interpret(result, "main", (0, 7))
                self.assertEqual(outcome.kind, "normal")
                self.assertEqual(outcome.value, 0)
                self.assertEqual(outcome.output, [])

                taken = _interpret(result, "main", (1, 7))
                self.assertEqual(taken.kind, "fault")
                self.assertEqual(taken.category, "division-by-zero")
                self.assertEqual(taken.output, [("emit", (0,))])
                # The fault stayed in the loop body block.
                self.assertEqual(taken.site[1], "b2")
                self.assertEqual(taken.site[2], operator)

    def test_loop_samples_match_baseline_under_licm_orders(self):
        orders = [
            ("ssa", "fold", "licm", "dce", "ssa"),
            ("ssa", "licm"),
            ("ssa", "fold", "licm", "licm"),
        ]
        for builder in (_loop_step_program, _loop_fault_program):
            lowered = lower_module(builder())
            # Discover the sample's entry and a few argument rows via the
            # shared case table indirectly: run both its fault and normal
            # inputs directly.
            if builder is _loop_step_program:
                entry, rows = "loopstep", [(0,), (5,), (20,)]
            else:
                entry, rows = "looptrap", [(2,), (5,)]
            for arguments in rows:
                baseline = _interpret(lowered, entry, arguments)
                for order in orders:
                    with self.subTest(
                            builder=builder.__name__,
                            arguments=arguments, order=order):
                        result = optimize_module(
                            copy.deepcopy(lowered), order)
                        self.assertEqual(
                            _interpret(result, entry, arguments),
                            baseline)


if __name__ == "__main__":
    unittest.main()
