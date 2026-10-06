"""Tests for SSA dead-code elimination (:func:`eliminate_dead_code`).

The tests pin down:

* the API contract (``TypeError`` for non-modules, ``ValueError`` for
  non-SSA modules, independent copies for empty/no-op modules);
* full independence from the input (no mutation, no shared functions,
  blocks, instructions or phi containers);
* roots: ``return``/``br`` operands are live; every ``call`` survives even
  with an unused result, and its arguments stay live;
* deletion of unreachable ``const``/``binop`` chains and of closed phi-only
  cycles, including the pure definitions that only feed them;
* preservation of function order, signatures, block labels/order,
  terminators, surviving instruction order and phi predecessor order, with
  SSA id holes left behind and no renumbering;
* idempotence, stable rendering and observable semantics (return value and
  call trace) via the shared IR interpreter.
"""
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
    Phi,
    Return,
    Slot,
    Temp,
    eliminate_dead_code,
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
from test_semantic_equivalence import _CASES, _BOOL_LITERAL_AST, _interpret


# --------------------------------------------------------------------------
# Traversal helpers
# --------------------------------------------------------------------------


def _all_instructions(fn_):
    for block in fn_.blocks:
        yield from block.instructions


def _all_values(fn_):
    values = []
    for block in fn_.blocks:
        for phi in block.phis:
            values.extend(phi.entries.values())
        for ins in block.instructions:
            if isinstance(ins, BinOp):
                values.extend([ins.left, ins.right])
            elif isinstance(ins, Call):
                values.extend(ins.args)
        term = block.terminator
        if isinstance(term, Return) and term.value is not None:
            values.append(term.value)
        elif isinstance(term, Branch):
            values.append(term.cond)
    return values


def _defined_ids(fn_):
    ids = {p.temp.id for p in fn_.params}
    for block in fn_.blocks:
        ids.update(phi.dest.id for phi in block.phis)
        ids.update(ins.dest.id for ins in block.instructions)
    return ids


def assert_closed_output(test, fn_):
    """Every surviving reference resolves inside the output function."""
    defined = {id(v) for v in [p.temp for p in fn_.params]}
    for block in fn_.blocks:
        defined.update(id(phi.dest) for phi in block.phis)
        defined.update(id(ins.dest) for ins in block.instructions)
    blocks = set(fn_.blocks)
    for block in fn_.blocks:
        for phi in block.phis:
            for source_block in phi.entries:
                test.assertIn(source_block, blocks)
        for value in _all_values(fn_):
            test.assertIn(id(value), defined)
    term = fn_.entry.terminator
    test.assertIsNotNone(term)


# --------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def test_returns_new_module_marked_ssa(self):
        ssa = to_ssa(lower_module(program(
            func("f", [param("a", "int")], "int",
                 [ret(arith("add", var("a"), int_(1)))])
        )))
        result = eliminate_dead_code(ssa)
        self.assertIsInstance(result, Module)
        self.assertIsNot(result, ssa)
        self.assertTrue(result.ssa)
        self.assertTrue(all(f.ssa for f in result.functions))

    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.assertRaises(TypeError):
                eliminate_dead_code(bad)

    def test_non_ssa_module_raises_value_error(self):
        non_ssa = lower_module(program(
            func("f", [param("a", "int")], "int", [ret(var("a"))])
        ))
        self.assertFalse(non_ssa.ssa)
        with self.assertRaises(ValueError):
            eliminate_dead_code(non_ssa)

    def test_empty_module_returns_independent_copy(self):
        empty = Module([], ssa=True)
        result = eliminate_dead_code(empty)
        self.assertIsNot(result, empty)
        self.assertTrue(result.ssa)
        self.assertEqual(result.functions, [])

    def test_noop_module_returns_independent_copy(self):
        ssa = to_ssa(lower_module(program(
            func("f", [param("a", "int")], "int", [ret(var("a"))])
        )))
        result = eliminate_dead_code(ssa)
        self.assertEqual(render_module(result), render_module(ssa))
        self.assertIsNot(result, ssa)
        # Nothing is shared at the container level.
        for old_fn, new_fn in zip(ssa.functions, result.functions):
            self.assertIsNot(old_fn, new_fn)
            for old_block, new_block in zip(old_fn.blocks, new_fn.blocks):
                self.assertIsNot(old_block, new_block)
                self.assertIsNot(old_block.phis, new_block.phis)
                self.assertIsNot(
                    old_block.instructions, new_block.instructions
                )
                for old_ins, new_ins in zip(
                    old_block.instructions, new_block.instructions
                ):
                    self.assertIsNot(old_ins, new_ins)
                for old_phi, new_phi in zip(
                    old_block.phis, new_block.phis
                ):
                    self.assertIsNot(old_phi, new_phi)
                    self.assertIsNot(old_phi.entries, new_phi.entries)

    def test_input_module_is_not_mutated(self):
        ssa = to_ssa(lower_module(program(
            func("f", [param("c", "bool")], "int",
                 [
                     let("a", "int", arith("add", int_(1), int_(2))),
                     if_(var("c"), [ret(int_(1))], [ret(int_(2))]),
                 ])
        )))
        before = render_module(ssa)
        eliminate_dead_code(ssa)
        self.assertEqual(render_module(ssa), before)


# --------------------------------------------------------------------------
# Liveness roots and deletion
# --------------------------------------------------------------------------


def _dead_chain_program():
    # `a` is computed (const, const, binop) but never read: the whole
    # pure-definition chain must disappear.
    return program(func(
        "f", [param("c", "bool")], "int",
        [
            let("a", "int", arith("add", int_(40), int_(2))),
            if_(var("c"), [ret(int_(1))], [ret(int_(2))]),
        ],
    ))


class EliminationTests(unittest.TestCase):
    def test_unreachable_const_binop_chain_is_removed(self):
        ssa = to_ssa(lower_module(_dead_chain_program()))
        result = eliminate_dead_code(ssa).functions[0]
        # No instruction computes the dead add, and the consts feeding it
        # vanish too: only the two return literals remain.
        surviving = list(_all_instructions(result))
        self.assertEqual([type(i) for i in surviving], [Const, Const])
        self.assertEqual(
            sorted(i.dest.type for i in surviving), ["int", "int"]
        )
        self.assertEqual({i.value for i in surviving}, {1, 2})

    def test_numbering_keeps_holes_without_renumbering(self):
        ssa = to_ssa(lower_module(_dead_chain_program())).functions[0]
        result = eliminate_dead_code(to_ssa(
            lower_module(_dead_chain_program())
        )).functions[0]
        before_ids = _defined_ids(ssa)
        after_ids = _defined_ids(result)
        # Only definitions were removed; every surviving id existed before
        # and at least one hole remains (the dead chain's id is skipped).
        self.assertTrue(after_ids < before_ids)
        self.assertTrue(after_ids)
        # The return values keep their original ids rather than being
        # renumbered to close the gap.
        for block in result.blocks:
            term = block.terminator
            if isinstance(term, Return) and term.value is not None:
                self.assertIn(term.value.id, after_ids)

    def test_call_with_unused_result_is_kept_with_its_arguments(self):
        module = lower_module(program(
            func("log", [param("n", "int")], "void",
                 [let("r", "int", call("record", [var("n")]))]),
            func("record", [param("x", "int")], "int", [ret(var("x"))]),
        ))
        ssa = to_ssa(module).functions[0]
        result = eliminate_dead_code(to_ssa(module)).functions[0]
        calls = [i for i in _all_instructions(result) if isinstance(i, Call)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "record")
        # The argument definition feeding the call survives as well.
        self.assertEqual(_defined_ids(result), {0, 1})
        assert_closed_output(self, result)
        # And the call is still observable: interpreting the optimized
        # module records it, in order, with its argument.
        optimized = eliminate_dead_code(to_ssa(module))
        value, trace = _interpret(optimized, "log", (7,))
        self.assertIsNone(value)
        self.assertEqual(trace, [("record", (7,))])

    def test_dead_phi_cycle_and_its_feeders_die_together(self):
        # Hand-built SSA: a loop header carries two phis p/q that only feed
        # each other on the back edge; neither result is read anywhere.
        # The consts supplying their entry edges die with the phis, while
        # the returned parameter stays live.
        param_temp = Temp(0, "int")
        c0, c1 = Temp(1, "int"), Temp(2, "int")
        p, q = Temp(3, "int"), Temp(4, "int")
        b0, b1, b2 = Block(0), Block(1), Block(2)
        b0.instructions = [Const(c0, 1), Const(c1, 2)]
        b0.terminator = Jump(b1)
        phi_p = Phi(p, {b0: c0, b2: q})
        phi_q = Phi(q, {b0: c1, b2: p})
        b1.phis = [phi_p, phi_q]
        b1.terminator = Return(param_temp)
        b2.terminator = Jump(b1)
        ssa = Module([Function(
            "cycle",
            [Parameter("x", Slot(0, "int"), param_temp)],
            "int", [], [b0, b1, b2], b0, ssa=True,
        )], ssa=True)

        result = eliminate_dead_code(ssa).functions[0]
        # All blocks and the terminator remain; both phis, both consts go.
        self.assertEqual([b.id for b in result.blocks], [0, 1, 2])
        self.assertEqual(
            [phi for b in result.blocks for phi in b.phis], []
        )
        self.assertEqual(
            [ins for b in result.blocks for ins in b.instructions], []
        )
        self.assertEqual(_defined_ids(result), {0})
        self.assertIs(result.blocks[1].terminator.value,
                      result.params[0].temp)
        assert_closed_output(self, result)

    def test_live_phi_keeps_predecessor_order_and_copied_blocks(self):
        ast = program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", int_(1)),
                if_(var("c"), [assign("a", var("x"))],
                    [assign("a", int_(2))]),
                ret(var("a")),
            ],
        ))
        result = eliminate_dead_code(to_ssa(lower_module(ast))).functions[0]
        merge = result.blocks[3]
        self.assertEqual(len(merge.phis), 1)
        phi = merge.phis[0]
        self.assertEqual(
            [b.label for b in phi.entries], ["b1", "b2"]
        )
        # Phi source blocks are copied block objects of the output module.
        for source in phi.entries:
            self.assertIn(source, result.blocks)
        self.assertIs(merge.terminator.value, phi.dest)
        assert_closed_output(self, result)

    def test_branch_condition_chain_is_live(self):
        ast = program(func(
            "f", [param("n", "int")], "void",
            [
                while_(
                    compare("gt", arith("add", var("n"), int_(0)), int_(0)),
                    [],
                ),
            ],
        ))
        result = eliminate_dead_code(to_ssa(lower_module(ast))).functions[0]
        kinds = [(i.kind, i.operator)
                 for i in _all_instructions(result)
                 if isinstance(i, BinOp)]
        self.assertIn(("arith", "add"), kinds)
        self.assertIn(("compare", "gt"), kinds)


# --------------------------------------------------------------------------
# Structural preservation
# --------------------------------------------------------------------------


COUNTER = func(
    "counter",
    [param("n", "int")],
    "int",
    [
        let("acc", "int", int_(0)),
        while_(
            compare("gt", var("n"), int_(0)),
            [
                assign("acc", arith("add", var("acc"), var("n"))),
                assign("n", arith("sub", var("n"), int_(1))),
            ],
        ),
        ret(var("acc")),
    ],
)


class PreservationTests(unittest.TestCase):
    def test_signature_blocks_terminators_preserved(self):
        ssa = to_ssa(lower_module(program(COUNTER))).functions[0]
        result = eliminate_dead_code(to_ssa(
            lower_module(program(COUNTER))
        )).functions[0]
        self.assertEqual(result.name, "counter")
        self.assertEqual(result.ret_type, "int")
        self.assertEqual(
            [(p.name, p.temp.type) for p in result.params],
            [("n", "int")],
        )
        self.assertEqual(
            [b.id for b in result.blocks], [b.id for b in ssa.blocks]
        )
        for old_block, new_block in zip(ssa.blocks, result.blocks):
            self.assertEqual(
                type(old_block.terminator), type(new_block.terminator)
            )
            old_t, new_t = old_block.terminator, new_block.terminator
            if isinstance(old_t, Jump):
                self.assertEqual(old_t.target.id, new_t.target.id)
            elif isinstance(old_t, Branch):
                self.assertEqual(
                    old_t.true_target.id, new_t.true_target.id
                )
                self.assertEqual(
                    old_t.false_target.id, new_t.false_target.id
                )

    def test_surviving_instructions_keep_relative_order(self):
        # A dead const sits between two live calls; the surviving pair must
        # keep its original order.
        module = lower_module(program(
            func("f", [param("n", "int")], "void",
                 [
                     let("a", "int", call("first", [var("n")])),
                     let("d", "int", int_(99)),
                     let("b", "int", arith(
                         "add", call("second", [var("n")]), int_(0))),
                 ]),
            func("first", [param("x", "int")], "int", [ret(var("x"))]),
            func("second", [param("x", "int")], "int", [ret(var("x"))]),
        ))
        result = eliminate_dead_code(to_ssa(module)).functions[0]
        names = [i.name for i in _all_instructions(result)
                 if isinstance(i, Call)]
        self.assertEqual(names, ["first", "second"])
        # The dead const is gone; the unused add feeding `b` and its const-0
        # operand are removed too, while the `second` call itself survives.
        self.assertNotIn(
            99, [i.value for i in _all_instructions(result)
                 if isinstance(i, Const)]
        )

    def test_function_order_preserved(self):
        module = lower_module(program(
            func("a", [], "void", []),
            func("b", [], "void",
                 [let("x", "int", int_(1))]),
            func("c", [], "void", []),
        ))
        result = eliminate_dead_code(to_ssa(module))
        self.assertEqual([f.name for f in result.functions],
                         ["a", "b", "c"])
        # The dead literal in b is removed but b itself is retained.
        self.assertEqual(
            list(_all_instructions(result.functions[1])), []
        )


# --------------------------------------------------------------------------
# Idempotence, determinism and observable semantics
# --------------------------------------------------------------------------


class IdempotenceTests(unittest.TestCase):
    PROGRAMS = [ast for _, ast, _, _ in _CASES] + [
        _BOOL_LITERAL_AST,
        _dead_chain_program(),
        program(COUNTER),
    ]

    def test_idempotent_structurally_and_textually(self):
        for ast in self.PROGRAMS:
            module = lower_module(ast)
            once = eliminate_dead_code(to_ssa(module))
            twice = eliminate_dead_code(once)
            self.assertEqual(
                render_module(once), render_module(twice)
            )
            for f1, f2 in zip(once.functions, twice.functions):
                self.assertEqual(
                    [b.id for b in f1.blocks], [b.id for b in f2.blocks]
                )
                self.assertEqual(
                    [len(b.phis) for b in f1.blocks],
                    [len(b.phis) for b in f2.blocks],
                )
                self.assertEqual(
                    _defined_ids(f1), _defined_ids(f2)
                )
            self.assertIsNot(twice, once)

    def test_rendering_is_stable(self):
        for ast in self.PROGRAMS:
            module = lower_module(ast)
            first = render_module(eliminate_dead_code(to_ssa(module)))
            second = render_module(eliminate_dead_code(to_ssa(module)))
            self.assertEqual(first, second)

    def test_observable_behavior_preserved(self):
        for name, ast, entry, runs in _CASES:
            for arguments, expected_return, expected_trace in runs:
                with self.subTest(case=name, arguments=arguments):
                    module = lower_module(ast)
                    ssa_result = _interpret(to_ssa(module), entry, arguments)
                    optimized = eliminate_dead_code(to_ssa(module))
                    opt_result = _interpret(optimized, entry, arguments)
                    self.assertEqual(opt_result, ssa_result)
                    self.assertEqual(
                        opt_result, (expected_return, expected_trace)
                    )

    def test_bool_literal_program_preserved(self):
        module = lower_module(_BOOL_LITERAL_AST)
        optimized = eliminate_dead_code(to_ssa(module))
        self.assertEqual(_interpret(optimized, "lit", ()), (True, []))


if __name__ == "__main__":
    unittest.main()
