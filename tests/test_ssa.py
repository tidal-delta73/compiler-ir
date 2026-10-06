"""Tests for the SSA conversion entry point ``to_ssa``."""
import unittest

from compiler_ir import (
    Module,
    emit_ir,
    lower_module,
    render_module,
    to_ssa,
)


def int_(value):
    return {"kind": "int", "value": value}


def bool_(value):
    return {"kind": "bool", "value": value}


def var(name):
    return {"kind": "var", "name": name}


def arith(op, left, right):
    return {"kind": "arith", "op": op, "left": left, "right": right}


def compare(op, left, right):
    return {"kind": "compare", "op": op, "left": left, "right": right}


def logical(op, left, right):
    return {"kind": "logical", "op": op, "left": left, "right": right}


def call(name, args):
    return {"kind": "call", "name": name, "args": args}


def let(name, typ, init):
    return {"kind": "let", "name": name, "type": typ, "init": init}


def assign(target, value):
    return {"kind": "assign", "target": target, "value": value}


def if_(cond, then, else_=None):
    return {"kind": "if", "cond": cond, "then": then, "else": else_ or []}


def while_(cond, body):
    return {"kind": "while", "cond": cond, "body": body}


def ret(value=None):
    return {"kind": "return", "value": value}


def block(body):
    return {"kind": "block", "body": body}


def func(name, params, ret_type, body):
    return {"name": name, "params": params, "ret_type": ret_type, "body": body}


def param(name, typ):
    return {"name": name, "type": typ}


def program(*funcs):
    return {"functions": list(funcs)}


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


class EntryPointTests(unittest.TestCase):
    def test_non_module_raises_type_error(self):
        for bad in (None, [], {}, "module", 42, object()):
            with self.assertRaises(TypeError, msg=repr(bad)):
                to_ssa(bad)

    def test_function_is_not_a_module(self):
        module = lower_module(program(func("f", [], "void", [])))
        with self.assertRaises(TypeError):
            to_ssa(module.functions[0])

    def test_returns_new_module_input_untouched(self):
        module = lower_module(program(COUNTER))
        before = render_module(module)
        ssa = to_ssa(module)
        self.assertIsNot(ssa, module)
        self.assertIsInstance(ssa, Module)
        self.assertIsNot(ssa.functions[0], module.functions[0])
        # The original module still renders exactly as before.
        self.assertEqual(render_module(module), before)
        self.assertIn("%v0", render_module(module))

    def test_ssa_module_has_no_slots(self):
        ssa = to_ssa(lower_module(program(COUNTER)))
        text = render_module(ssa)
        self.assertNotIn("%v", text)
        self.assertEqual(ssa.functions[0].locals, [])


class IfElseTests(unittest.TestCase):
    IF_ELSE = func(
        "f",
        [param("c", "bool")],
        "int",
        [
            let("x", "int", int_(1)),
            if_(var("c"), [assign("x", int_(2))], [assign("x", int_(3))]),
            ret(var("x")),
        ],
    )

    def test_if_else_merge_phi_full_text(self):
        text = render_module(to_ssa(lower_module(program(self.IF_ELSE))))
        self.assertEqual(
            text,
            "module\n"
            "\n"
            "function f(c: bool @ %t0) -> int {\n"
            "  b0:\n"
            "    %t1: int = const 1\n"
            "    br %t0, b1, b2\n"
            "  b1:\n"
            "    %t2: int = const 2\n"
            "    jump b3\n"
            "  b2:\n"
            "    %t3: int = const 3\n"
            "    jump b3\n"
            "  b3:\n"
            "    %t4: int = phi [b1: %t2, b2: %t3]\n"
            "    return %t4\n"
            "}\n",
        )

    def test_phi_line_has_type_labels_and_values(self):
        text = render_module(to_ssa(lower_module(program(self.IF_ELSE))))
        self.assertIn("%t4: int = phi [b1: %t2, b2: %t3]", text)

    def test_same_value_on_all_edges_leaves_no_phi(self):
        # Neither branch writes x: the merge sees one reaching definition.
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "int",
                [let("x", "int", int_(1)), if_(var("c"), [], []), ret(var("x"))],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertNotIn("phi", text)
        self.assertIn("return %t1", text)

    def test_identical_reaching_value_leaves_no_phi(self):
        # Both branches assign x from the same source value.
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "int",
                [
                    let("y", "int", int_(1)),
                    let("x", "int", int_(0)),
                    if_(
                        var("c"),
                        [assign("x", var("y"))],
                        [assign("x", var("y"))],
                    ),
                    ret(var("x")),
                ],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertNotIn("phi", text)

    def test_unused_merge_value_leaves_no_phi(self):
        # x is written on both branches but never read afterwards.
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "int",
                [
                    let("x", "int", int_(1)),
                    if_(var("c"), [assign("x", int_(2))], [assign("x", int_(3))]),
                    ret(int_(0)),
                ],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertNotIn("phi", text)

    def test_both_branches_return_no_merge_no_phi(self):
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "int",
                [if_(var("c"), [ret(int_(1))], [ret(int_(2))])],
            )
        )
        ssa = to_ssa(lower_module(ast))
        self.assertEqual(len(ssa.functions[0].blocks), 3)
        self.assertNotIn("phi", render_module(ssa))

    def test_one_branch_returns_single_predecessor_no_phi(self):
        # The merge has a single reachable predecessor: no phi is needed.
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "int",
                [
                    let("x", "int", int_(1)),
                    if_(var("c"), [ret(int_(0))], []),
                    ret(var("x")),
                ],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertNotIn("phi", text)

    def test_nested_if_renaming(self):
        ast = program(
            func(
                "f",
                [param("a", "bool"), param("b", "bool")],
                "int",
                [
                    let("x", "int", int_(0)),
                    if_(
                        var("a"),
                        [if_(var("b"), [assign("x", int_(1))], [assign("x", int_(2))])],
                        [assign("x", int_(3))],
                    ),
                    ret(var("x")),
                ],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        # Inner merge joins the two inner assignments; outer merge joins
        # the inner merge with the else branch.
        self.assertIn("%t6: int = phi [b3: %t4, b4: %t5]", text)
        self.assertIn("%t7: int = phi [b2: %t3, b5: %t6]", text)
        self.assertIn("return %t7", text)

    def test_nested_block_scope_renaming(self):
        ast = program(
            func(
                "f",
                [],
                "int",
                [
                    let("x", "int", int_(1)),
                    block([let("x", "int", int_(2)), assign("x", int_(3))]),
                    ret(var("x")),
                ],
            )
        )
        # The inner shadowing x is distinct from the outer one returned.
        text = render_module(to_ssa(lower_module(ast)))
        self.assertIn("return %t0", text)
        self.assertNotIn("phi", text)


class LoopTests(unittest.TestCase):
    def test_while_loop_header_phi(self):
        text = render_module(to_ssa(lower_module(program(COUNTER))))
        # Loop-carried variables take the entry value from b0 and the
        # back-edge value from the body block b2.
        self.assertIn("%t2: int = phi [b0: %t0, b2: %t8]", text)
        self.assertIn("%t3: int = phi [b0: %t1, b2: %t6]", text)
        self.assertIn("return %t3", text)

    def test_while_full_text(self):
        text = render_module(to_ssa(lower_module(program(COUNTER))))
        self.assertEqual(
            text,
            "module\n"
            "\n"
            "function counter(n: int @ %t0) -> int {\n"
            "  b0:\n"
            "    %t1: int = const 0\n"
            "    jump b1\n"
            "  b1:\n"
            "    %t2: int = phi [b0: %t0, b2: %t8]\n"
            "    %t3: int = phi [b0: %t1, b2: %t6]\n"
            "    %t4: int = const 0\n"
            "    %t5: bool = compare gt %t2 %t4\n"
            "    br %t5, b2, b3\n"
            "  b2:\n"
            "    %t6: int = arith add %t3 %t2\n"
            "    %t7: int = const 1\n"
            "    %t8: int = arith sub %t2 %t7\n"
            "    jump b1\n"
            "  b3:\n"
            "    return %t3\n"
            "}\n",
        )

    def test_loop_invariant_variable_gets_no_phi(self):
        # y is read inside the loop but never written: it reaches the
        # header unchanged from every predecessor, so no phi survives.
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "int",
                [
                    let("y", "int", int_(5)),
                    let("i", "int", int_(0)),
                    while_(
                        var("c"),
                        [assign("i", arith("add", var("i"), var("y")))],
                    ),
                    ret(var("i")),
                ],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        phis = [l.strip() for l in text.splitlines() if " phi " in l]
        self.assertEqual(len(phis), 1)  # only the loop-carried i
        self.assertIn("= phi [b0: %t2, b2: %t4]", phis[0])

    def test_unchanged_param_in_loop_phi_is_trivial(self):
        # The condition parameter is read in the loop header but never
        # written: its would-be phi has the same value on every edge.
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "void",
                [while_(var("c"), [])],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertNotIn("phi", text)

    def test_body_always_returns_no_back_edge(self):
        # The back edge does not exist, so the header has one predecessor
        # and needs no phi.
        ast = program(
            func(
                "f",
                [param("c", "bool")],
                "int",
                [
                    let("x", "int", int_(1)),
                    while_(var("c"), [ret(var("x"))]),
                    ret(var("x")),
                ],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertNotIn("phi", text)


class ShortCircuitTests(unittest.TestCase):
    def test_and_multi_path_write_is_split(self):
        ast = program(
            func(
                "f",
                [param("a", "bool"), param("b", "bool")],
                "bool",
                [ret(logical("and", var("a"), var("b")))],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertEqual(
            text,
            "module\n"
            "\n"
            "function f(a: bool @ %t0, b: bool @ %t1) -> bool {\n"
            "  b0:\n"
            "    br %t0, b1, b2\n"
            "  b1:\n"
            "    %t2: bool = copy %t1\n"
            "    jump b3\n"
            "  b2:\n"
            "    %t3: bool = const false\n"
            "    jump b3\n"
            "  b3:\n"
            "    %t4: bool = phi [b1: %t2, b2: %t3]\n"
            "    return %t4\n"
            "}\n",
        )

    def test_or_multi_path_write_is_split(self):
        ast = program(
            func(
                "f",
                [param("a", "bool"), param("b", "bool")],
                "bool",
                [ret(logical("or", var("a"), var("b")))],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        self.assertIn("%t2: bool = const true", text)
        self.assertIn("%t3: bool = copy %t1", text)
        self.assertIn("%t4: bool = phi [b1: %t2, b2: %t3]", text)
        self.assertIn("return %t4", text)

    def test_short_circuit_inside_if_condition(self):
        # The logical operator sits in the if condition: its merge block
        # joins the two short-circuit paths before the if branches.
        ast = program(
            func(
                "f",
                [param("a", "bool"), param("b", "bool")],
                "int",
                [
                    let("x", "int", int_(0)),
                    if_(
                        logical("and", var("a"), var("b")),
                        [assign("x", int_(1))],
                        [],
                    ),
                    ret(var("x")),
                ],
            )
        )
        text = render_module(to_ssa(lower_module(ast)))
        # Phi for the short-circuit result at the logical merge (b3)...
        self.assertIn("%t5: bool = phi [b1: %t3, b2: %t4]", text)
        self.assertIn("br %t5, b4, b5", text)
        # ...and a phi for x at the if merge (b6), where the untouched
        # else path carries the value from before the if.
        self.assertIn("%t7: int = phi [b4: %t6, b5: %t2]", text)
        self.assertIn("return %t7", text)
        # Every temp is defined exactly once.
        self._assert_ssa_invariant(text)

    def _assert_ssa_invariant(self, text):
        import re

        defs = re.findall(r"^\s+(%t\d+):", text, flags=re.MULTILINE)
        self.assertEqual(len(defs), len(set(defs)))


class CallAndOrderTests(unittest.TestCase):
    def test_calls_and_side_effect_order_preserved(self):
        ast = program(
            func("g", [param("x", "int")], "int", [ret(var("x"))]),
            func(
                "f",
                [param("c", "bool")],
                "int",
                [
                    let("a", "int", call("g", [int_(1)])),
                    if_(
                        var("c"),
                        [assign("a", call("g", [int_(2)]))],
                        [assign("a", call("g", [int_(3)]))],
                    ),
                    ret(call("g", [var("a")])),
                ],
            ),
        )
        text = render_module(to_ssa(lower_module(ast)))
        calls = [i for i, l in enumerate(text.splitlines()) if "call g(" in l]
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls, sorted(calls))
        # The merge phi selects between the two branch call results.
        self.assertIn("%t7: int = phi [b1: %t4, b2: %t6]", text)
        # The final call passes the phi result.
        self.assertIn("call g(%t7)", text)

    def test_function_and_block_order_preserved(self):
        ast = program(COUNTER, func("noop", [], "void", []))
        module = lower_module(ast)
        ssa = to_ssa(module)
        self.assertEqual(
            [f.name for f in ssa.functions],
            [f.name for f in module.functions],
        )
        for old, new in zip(module.functions, ssa.functions):
            self.assertEqual(
                [b.id for b in new.blocks], [b.id for b in old.blocks]
            )
            self.assertEqual(new.entry.id, old.entry.id)
            self.assertEqual(new.ret_type, old.ret_type)

    def test_void_function_and_bare_return(self):
        text = render_module(to_ssa(lower_module(program(func("noop", [], "void", [])))))
        self.assertEqual(
            text,
            "module\n\nfunction noop() -> void {\n  b0:\n    return\n}\n",
        )


class DeterminismTests(unittest.TestCase):
    def test_repeated_conversion_is_byte_identical(self):
        ast = program(COUNTER)
        first = render_module(to_ssa(lower_module(ast)))
        second = render_module(to_ssa(lower_module(ast)))
        self.assertEqual(first, second)

    def test_idempotent_fixed_point(self):
        ssa = to_ssa(lower_module(program(COUNTER)))
        again = to_ssa(ssa)
        self.assertEqual(render_module(again), render_module(ssa))
        # No additional phis and no renumbering on the second pass.
        count = render_module(ssa).count(" phi ")
        self.assertEqual(render_module(again).count(" phi "), count)

    def test_numbering_starts_from_zero_per_function(self):
        import re

        ast = program(
            COUNTER,
            func("g", [param("x", "int")], "int", [ret(var("x"))]),
        )
        text = render_module(to_ssa(lower_module(ast)))
        for body in text.split("function ")[1:]:
            # Definitions: parameters (@ %tN) and phi/instruction results.
            ids = [int(m) for m in re.findall(r"@ %t(\d+)", body)]
            ids += [int(m) for m in re.findall(r"%t(\d+):", body)]
            self.assertEqual(sorted(ids), list(range(len(ids))), body)

    def test_non_ssa_render_unchanged_by_ssa_support(self):
        # render_module output for the lowered (non-SSA) module is exactly
        # what the pipeline always produced.
        ast = program(COUNTER)
        self.assertEqual(render_module(lower_module(ast)), emit_ir(ast))


if __name__ == "__main__":
    unittest.main()
