"""Tests for SSA dead-code elimination (:func:`eliminate_dead_code`).

The pass is tested five ways:

* the API contract -- new SSA module, input never mutated, ``TypeError``
  for non-modules, ``ValueError`` for non-SSA modules, independent copies
  for the empty module and already-minimal modules;
* root seeding -- ``Call`` instructions (and their argument chains) and
  ``Return``/``Branch`` operands survive, while pure ``Const``/``BinOp``
  chains feeding nothing observable disappear;
* phi handling -- dead phis pinned in ``to_ssa`` output only by dead
  instructions, and loop-carried phi cycles, are removed whole, whereas
  live phis keep their entries and predecessor order;
* preservation -- function/block/terminator/order are untouched, surviving
  SSA numbers keep their ids (holes allowed), and every retained reference
  resolves inside the output module;
* semantics -- the shared IR interpreter executes the eliminated module
  and observes the same return value and call sequence as the input.
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
from test_semantic_equivalence import _CASES, _interpret


# --------------------------------------------------------------------------
# Structural helpers
# --------------------------------------------------------------------------


def _edges(block):
    term = block.terminator
    if isinstance(term, Jump):
        return [term.target]
    if isinstance(term, Branch):
        return [term.true_target, term.false_target]
    return []


def _operands(ins):
    if isinstance(ins, BinOp):
        return [ins.left, ins.right]
    if isinstance(ins, Call):
        return list(ins.args)
    return []


def _term_operands(term):
    if isinstance(term, Return):
        return [] if term.value is None else [term.value]
    if isinstance(term, Branch):
        return [term.cond]
    return []


def _all_defs(fn_):
    defs = [p.temp for p in fn_.params]
    for block in fn_.blocks:
        defs.extend(phi.dest for phi in block.phis)
        defs.extend(ins.dest for ins in block.instructions)
    return defs


def _all_uses(fn_):
    uses = []
    for block in fn_.blocks:
        for phi in block.phis:
            uses.extend(phi.entries.values())
        for ins in block.instructions:
            uses.extend(_operands(ins))
        uses.extend(_term_operands(block.terminator))
    return uses


def _container_objects(module):
    """All mutable container nodes reachable from ``module``."""
    types = (Function, Block, Phi, Const, BinOp, Call, Return, Jump,
             Branch, Parameter)
    found = set()
    stack = [module]
    while stack:
        obj = stack.pop()
        if id(obj) in found:
            continue
        if isinstance(obj, types):
            found.add(id(obj))
        for attr in vars(obj).values() if hasattr(obj, "__dict__") else ():
            if isinstance(attr, types):
                stack.append(attr)
            elif isinstance(attr, list):
                stack.extend(x for x in attr if isinstance(x, types))
            elif isinstance(attr, dict):
                stack.extend(x for x in attr.values() if isinstance(x, types))
    return found


def assert_ssa_closed(test, module):
    """Every retained use, edge target and phi source lives in the output."""
    for fn_ in module.functions:
        def_ids = {id(d) for d in _all_defs(fn_)}
        block_ids = {id(b) for b in fn_.blocks}
        test.assertIn(id(fn_.entry), block_ids)
        for block in fn_.blocks:
            for phi in block.phis:
                for pred, value in phi.entries.items():
                    test.assertIn(id(pred), block_ids,
                                  "phi source block outside output module")
                    test.assertIn(id(value), def_ids,
                                  "phi incoming value undefined in output")
            for ins in block.instructions:
                for value in _operands(ins):
                    test.assertIn(id(value), def_ids,
                                  f"operand of {ins.op} undefined in output")
            for value in _term_operands(block.terminator):
                test.assertIn(id(value), def_ids,
                              "terminator operand undefined in output")
            for target in _edges(block):
                test.assertIn(id(target), block_ids,
                              "terminator targets block outside output module")


# --------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def test_returns_new_module_marked_ssa(self):
        ssa = to_ssa(lower_module(_dead_chain_ast()))
        result = eliminate_dead_code(ssa)
        self.assertIsInstance(result, Module)
        self.assertIsNot(result, ssa)
        self.assertTrue(result.ssa)
        self.assertTrue(all(f.ssa for f in result.functions))

    def test_input_module_is_not_mutated(self):
        ssa = to_ssa(lower_module(_dead_chain_ast()))
        before = render_module(ssa)
        eliminate_dead_code(ssa)
        self.assertEqual(render_module(ssa), before)

    def test_shares_no_mutable_containers_with_input(self):
        ssa = to_ssa(lower_module(_dead_chain_ast()))
        result = eliminate_dead_code(ssa)
        self.assertTrue(
            _container_objects(ssa).isdisjoint(_container_objects(result))
        )

    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.assertRaises(TypeError):
                eliminate_dead_code(bad)

    def test_non_ssa_module_raises_value_error(self):
        non_ssa = lower_module(_dead_chain_ast())
        self.assertFalse(non_ssa.ssa)
        with self.assertRaises(ValueError):
            eliminate_dead_code(non_ssa)
        # An explicitly empty non-SSA module is still the wrong flavor.
        with self.assertRaises(ValueError):
            eliminate_dead_code(Module([], ssa=False))

    def test_empty_module_returns_independent_copy(self):
        empty = to_ssa(lower_module(program()))
        self.assertEqual(empty.functions, [])
        result = eliminate_dead_code(empty)
        self.assertIsNot(result, empty)
        self.assertTrue(result.ssa)
        self.assertEqual(result.functions, [])
        self.assertEqual(render_module(result), render_module(empty))

    def test_minimal_module_returns_independent_equivalent_copy(self):
        ssa = to_ssa(lower_module(program(_COUNTER_LIKE)))
        once = eliminate_dead_code(ssa)
        twice = eliminate_dead_code(once)
        self.assertIsNot(twice, once)
        self.assertEqual(render_module(twice), render_module(once))
        self.assertTrue(
            _container_objects(once).isdisjoint(_container_objects(twice))
        )


# --------------------------------------------------------------------------
# Liveness roots and deletion
# --------------------------------------------------------------------------


# void f(x): pure dead chain, an observable call whose result feeds only a
# dead multiply, then nothing.
_DEAD_CHAIN_FUNC = func(
    "f", [param("x", "int")], "void",
    [
        let("d", "int", arith("add", int_(1), int_(2))),
        let("y", "int",
            call("keep", [arith("add", var("x"), int_(10))])),
        let("z", "int", arith("mul", var("y"), int_(3))),
    ],
)


def _keep_function():
    return func("keep", [param("v", "int")], "int", [ret(var("v"))])


def _dead_chain_module():
    return to_ssa(lower_module(
        program(_DEAD_CHAIN_FUNC, _keep_function())
    ))


def _dead_chain_ast():
    return program(_DEAD_CHAIN_FUNC, _keep_function())


class LivenessTests(unittest.TestCase):
    def test_pure_dead_constants_and_binop_chain_removed(self):
        result = eliminate_dead_code(_dead_chain_module()).functions[0]
        ops = [(type(i), getattr(i, "operator", None))
               for b in result.blocks for i in b.instructions]
        names = [i.name for b in result.blocks for i in b.instructions
                 if isinstance(i, Call)]
        # The dead 1+2 add, the y*3 multiply, and the constants 1,2,3 go;
        # the observable call stays, as do its argument add and constant 10.
        self.assertEqual(names, ["keep"])
        self.assertIn((BinOp, "add"), ops)
        self.assertNotIn((BinOp, "mul"), ops)
        consts = [i.value for b in result.blocks for i in b.instructions
                  if isinstance(i, Const)]
        self.assertIn(10, consts)
        self.assertEqual(set(consts) - {10}, set())

    def test_unused_call_is_retained_with_argument_chain(self):
        fn_ = eliminate_dead_code(_dead_chain_module()).functions[0]
        calls = [i for b in fn_.blocks for i in b.instructions
                 if isinstance(i, Call)]
        self.assertEqual(len(calls), 1)
        call_ins = calls[0]
        self.assertEqual(call_ins.name, "keep")
        # Its single argument is the surviving x+10 add result.
        arg = call_ins.args[0]
        owner = next(
            i for b in fn_.blocks for i in b.instructions
            if isinstance(i, BinOp) and i.dest is arg
        )
        self.assertEqual(owner.operator, "add")

    def test_branch_condition_chain_is_live(self):
        module = to_ssa(lower_module(program(func(
            "f", [param("x", "int")], "void",
            [
                let("d", "int", arith("add", int_(1), int_(2))),
                if_(compare("gt", var("x"), int_(0)),
                    [let("e", "int", int_(9))]),
            ],
        ))))
        fn_ = eliminate_dead_code(module).functions[0]
        # The branch compare and the constant 0 survive; dead add and
        # branch-local const do not.
        compares = [i for b in fn_.blocks for i in b.instructions
                    if isinstance(i, BinOp) and i.kind == "compare"]
        self.assertEqual(len(compares), 1)
        consts = {i.value for b in fn_.blocks for i in b.instructions
                  if isinstance(i, Const)}
        self.assertIn(0, consts)
        self.assertNotIn(1, consts)
        self.assertNotIn(9, consts)
        self.assertIsInstance(fn_.blocks[0].terminator, Branch)

    def test_returned_chain_is_live_dead_sibling_removed(self):
        module = to_ssa(lower_module(program(func(
            "f", [param("x", "int")], "int",
            [
                let("d", "int", arith("add", int_(1), int_(2))),
                ret(arith("mul", var("x"), int_(4))),
            ],
        ))))
        fn_ = eliminate_dead_code(module).functions[0]
        ops = [i for b in fn_.blocks for i in b.instructions
               if isinstance(i, BinOp)]
        self.assertEqual([i.operator for i in ops], ["mul"])
        consts = {i.value for b in fn_.blocks for i in b.instructions
                  if isinstance(i, Const)}
        self.assertEqual(consts, {4})


# --------------------------------------------------------------------------
# Phi elimination
# --------------------------------------------------------------------------


def _dead_phi_program():
    # `a` merges in the branches and is read only by an unused add `u`;
    # to_ssa keeps the merge phi because that add reads it, while DCE must
    # remove the phi together with the whole dead chain.
    return program(func(
        "f", [param("c", "bool"), param("x", "int")], "int",
        [
            let("a", "int", int_(1)),
            if_(var("c"),
                [assign("a", arith("add", var("x"), int_(2)))],
                [assign("a", arith("mul", var("x"), int_(3)))]),
            let("u", "int", arith("add", var("a"), int_(0))),
            ret(int_(7)),
        ],
    ))


def _dead_loop_phi_program():
    # `j` is carried around the loop but never read after it; the loop
    # counter `n` stays because the branch condition reads it.
    return program(func(
        "f", [param("n", "int")], "void",
        [
            let("j", "int", int_(0)),
            while_(compare("gt", var("n"), int_(0)),
                   [
                       assign("j", arith("add", var("j"), int_(1))),
                       assign("n", arith("sub", var("n"), int_(1))),
                   ]),
        ],
    ))


class PhiEliminationTests(unittest.TestCase):
    def test_phi_only_used_by_dead_instruction_is_removed(self):
        ssa = to_ssa(lower_module(_dead_phi_program()))
        # Sanity: to_ssa output does carry the merge phi plus the dead add.
        self.assertTrue(any(
            phi for fn_ in ssa.functions for b in fn_.blocks for phi in b.phis
        ))
        result = eliminate_dead_code(ssa).functions[0]
        self.assertFalse(any(b.phis for b in result.blocks))
        self.assertFalse(any(
            isinstance(i, BinOp) for b in result.blocks
            for i in b.instructions
        ))
        # Only the returned constant remains.
        self.assertEqual(
            [i.value for b in result.blocks for i in b.instructions
             if isinstance(i, Const)],
            [7],
        )

    def test_closed_loop_phi_cycle_removed_but_counter_phi_kept(self):
        ssa = to_ssa(lower_module(_dead_loop_phi_program()))
        header = ssa.functions[0].blocks[1]
        self.assertEqual(len(header.phis), 2)  # j and n before DCE
        result = eliminate_dead_code(ssa).functions[0]
        result_header = result.blocks[1]
        self.assertEqual(len(result_header.phis), 1)
        # The surviving phi is the counter's, whose entry value is the
        # parameter and whose back edge is the subtract -- both retained.
        phi = result_header.phis[0]
        by_name = {p.name: p.temp for p in result.params}
        self.assertIs(phi.entries[result.blocks[0]], by_name["n"])
        back = phi.entries[result.blocks[2]]
        sub = next(i for i in result.blocks[2].instructions
                   if isinstance(i, BinOp) and i.operator == "sub")
        self.assertIs(back, sub.dest)
        # The body holds exactly the counter's const 1 and subtract; the
        # j-increment add and its own const 1 disappeared.
        body = result.blocks[2]
        self.assertEqual(
            [type(i).__name__ for i in body.instructions],
            ["Const", "BinOp"],
        )
        self.assertEqual(body.instructions[0].value, 1)
        self.assertEqual(
            [i.operator for i in body.instructions if isinstance(i, BinOp)],
            ["sub"],
        )
        total_ones = sum(
            1 for b in result.blocks for i in b.instructions
            if isinstance(i, Const) and i.value == 1
        )
        self.assertEqual(total_ones, 1)

    def test_live_phi_keeps_entries_and_predecessor_order(self):
        # A merge value that really is returned: the phi survives and its
        # source blocks/values point into the output in label order.
        module = to_ssa(lower_module(program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", int_(1)),
                if_(var("c"),
                    [assign("a", arith("add", var("x"), int_(2)))],
                    [assign("a", arith("mul", var("x"), int_(3)))]),
                ret(var("a")),
            ],
        ))))
        result = eliminate_dead_code(module).functions[0]
        merge = result.blocks[3]
        self.assertEqual(len(merge.phis), 1)
        phi = merge.phis[0]
        keys = list(phi.entries.keys())
        self.assertEqual([k.id for k in keys], sorted(k.id for k in keys))
        self.assertEqual([k.label for k in keys], ["b1", "b2"])
        self.assertIs(merge.terminator.value, phi.dest)


# --------------------------------------------------------------------------
# Structural preservation
# --------------------------------------------------------------------------


_COUNTER_LIKE = func(
    "counter", [param("n", "int")], "int",
    [
        let("acc", "int", int_(0)),
        while_(compare("gt", var("n"), int_(0)),
               [
                   assign("acc", arith("add", var("acc"), var("n"))),
                   assign("n", arith("sub", var("n"), int_(1))),
               ]),
        ret(var("acc")),
    ],
)


class PreservationTests(unittest.TestCase):
    def test_function_order_signature_and_blocks_preserved(self):
        ssa = to_ssa(lower_module(program(_DEAD_CHAIN_FUNC, _keep_function())))
        result = eliminate_dead_code(ssa)
        self.assertEqual([f.name for f in result.functions], ["f", "keep"])
        for old, new in zip(ssa.functions, result.functions):
            self.assertEqual(new.ret_type, old.ret_type)
            self.assertEqual([p.name for p in new.params],
                             [p.name for p in old.params])
            self.assertEqual([b.id for b in new.blocks],
                             [b.id for b in old.blocks])
            for old_b, new_b in zip(old.blocks, new.blocks):
                self.assertEqual(type(new_b.terminator),
                                 type(old_b.terminator))

    def test_live_instruction_relative_order_preserved(self):
        ssa = to_ssa(lower_module(program(_DEAD_CHAIN_FUNC,
                                          _keep_function())))
        result = eliminate_dead_code(ssa).functions[0]
        for old_b, new_b in zip(ssa.functions[0].blocks, result.blocks):
            surviving = {i.dest.id for i in new_b.instructions}
            old_live = [i for i in old_b.instructions
                        if i.dest.id in surviving]
            self.assertEqual(
                [type(i) for i in old_live],
                [type(i) for i in new_b.instructions],
            )

    def test_surviving_numbers_keep_their_ids_with_holes(self):
        ssa = to_ssa(lower_module(program(func(
            "f", [param("x", "int")], "int",
            [
                let("d", "int", arith("add", int_(1), int_(2))),
                ret(int_(5)),
            ],
        ))))
        before = {d.id for d in _all_defs(ssa.functions[0])}
        result = eliminate_dead_code(ssa).functions[0]
        after = {d.id for d in _all_defs(result)}
        self.assertTrue(after.issubset(before))
        # %1, %2 (the dead constants) and %3 (the dead add) are gone;
        # the returned %4 keeps its number, leaving a real hole.
        self.assertEqual(after, {0, 4})

    def test_output_is_ssa_closed(self):
        for ast in (_dead_phi_program(), _dead_loop_phi_program(),
                    program(_DEAD_CHAIN_FUNC, _keep_function())):
            result = eliminate_dead_code(to_ssa(lower_module(ast)))
            assert_ssa_closed(self, result)


# --------------------------------------------------------------------------
# Fixpoint, idempotence and determinism
# --------------------------------------------------------------------------


class IdempotenceTests(unittest.TestCase):
    PROGRAMS = [
        _dead_phi_program(),
        _dead_loop_phi_program(),
        program(_DEAD_CHAIN_FUNC, _keep_function()),
        program(_COUNTER_LIKE),
    ]

    def test_repeated_elimination_is_stable(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            once = eliminate_dead_code(ssa)
            twice = eliminate_dead_code(once)
            self.assertEqual(render_module(once), render_module(twice))
            for f1, f2 in zip(once.functions, twice.functions):
                self.assertEqual(
                    [d.id for d in _all_defs(f1)],
                    [d.id for d in _all_defs(f2)],
                )
                self.assertEqual(
                    [len(b.phis) for b in f1.blocks],
                    [len(b.phis) for b in f2.blocks],
                )
            # A third application must change nothing either.
            thrice = eliminate_dead_code(twice)
            self.assertEqual(render_module(twice), render_module(thrice))

    def test_render_is_deterministic(self):
        for ast in self.PROGRAMS:
            ssa = to_ssa(lower_module(ast))
            self.assertEqual(
                render_module(eliminate_dead_code(ssa)),
                render_module(eliminate_dead_code(ssa)),
            )


# --------------------------------------------------------------------------
# Observable semantics: return value and call sequence are unchanged
# --------------------------------------------------------------------------


def _dead_code_rich_program():
    return program(
        func("rich", [param("n", "int")], "int", [
            let("d", "int", arith("add", int_(1), int_(2))),
            let("y", "int",
                call("keep", [arith("add", var("n"), int_(10))])),
            let("z", "int", arith("mul", var("y"), int_(3))),
            if_(compare("gt", var("n"), int_(0)),
                [let("a", "int",
                     arith("add", var("n"), call("keep", [int_(1)])))],
                []),
            ret(call("keep", [var("z")])),
        ]),
        _keep_function(),
    )


class SemanticPreservationTests(unittest.TestCase):
    def test_existing_cases_keep_return_and_call_trace(self):
        for name, ast, entry, runs in _CASES:
            for arguments, _expected_return, _expected_trace in runs:
                with self.subTest(case=name, arguments=arguments):
                    module = lower_module(ast)
                    expected = _interpret(module, entry, arguments)
                    eliminated = eliminate_dead_code(to_ssa(module))
                    self.assertEqual(
                        _interpret(eliminated, entry, arguments),
                        expected,
                    )

    def test_dead_code_rich_program(self):
        module = lower_module(_dead_code_rich_program())
        for arguments in ((0,), (4,), (-7,)):
            expected = _interpret(module, "rich", arguments)
            eliminated = eliminate_dead_code(to_ssa(module))
            assert_ssa_closed(self, eliminated)
            self.assertEqual(
                _interpret(eliminated, "rich", arguments), expected
            )

    def test_dead_phi_and_loop_programs_terminate_the_same(self):
        for ast, entry, args in (
            (_dead_phi_program(), "f", (True, 5)),
            (_dead_phi_program(), "f", (False, -2)),
            (_dead_loop_phi_program(), "f", (0,)),
            (_dead_loop_phi_program(), "f", (4,)),
        ):
            module = lower_module(ast)
            expected = _interpret(module, entry, args)
            eliminated = eliminate_dead_code(to_ssa(module))
            self.assertEqual(_interpret(eliminated, entry, args), expected)


if __name__ == "__main__":
    unittest.main()
