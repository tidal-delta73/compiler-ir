"""Tests for AST validation, semantic analysis and lowering."""

import unittest

import compiler_ir as cir
from compiler_ir import errors


def intlit(value):
    return {"kind": "int", "value": value}


def boollit(value):
    return {"kind": "bool", "value": value}


def var(name):
    return {"kind": "var", "name": name}


def binop(op, left, right):
    return {"kind": "binary", "op": op, "left": left, "right": right}


def call(name, *args):
    return {"kind": "call", "name": name, "args": list(args)}


def vdecl(name, type_name, init=None):
    node = {"kind": "var_decl", "name": name, "type": type_name}
    if init is not None:
        node["init"] = init
    return node


def assign(name, value):
    return {"kind": "assign", "target": name, "value": value}


def return_(value=None):
    node = {"kind": "return"}
    if value is not None:
        node["value"] = value
    return node


def if_(cond, then, else_=None):
    node = {"kind": "if", "cond": cond, "then": then}
    if else_ is not None:
        node["else"] = else_
    return node


def while_(cond, body):
    return {"kind": "while", "cond": cond, "body": body}


def block(body):
    return {"kind": "block", "body": body}


def func(name, params, return_type, body):
    return {
        "kind": "function",
        "name": name,
        "params": [{"name": p, "type": t} for p, t in params],
        "return_type": return_type,
        "body": body,
    }


def program(*functions):
    return {"functions": list(functions)}


class LoweringSmokeTest(unittest.TestCase):
    def test_simple_return_expression(self):
        ast = program(
            func("add1", [("x", "int")], "int",
                 [return_(binop("+", var("x"), intlit(1)))])
        )
        module = cir.lower_module(ast)
        self.assertIsInstance(module, cir.Module)
        self.assertEqual(len(module.functions), 1)
        fn = module.functions[0]
        self.assertEqual(fn.name, "add1")
        self.assertEqual([(p.name, p.type) for p in fn.params],
                         [("x", "int")])
        self.assertEqual(fn.return_type, "int")
        # Entry block, three expression temps and a return terminator.
        self.assertEqual(len(fn.blocks), 1)
        entry = fn.entry
        self.assertEqual([i.op for i in entry.instructions],
                         ["load", "const", "add"])
        self.assertEqual(entry.terminator.op, "return")
        self.assertEqual(entry.terminator.value.type, "int")

    def test_void_empty_body(self):
        module = cir.lower_module(program(func("noop", [], "void", [])))
        block = module.functions[0].entry
        self.assertEqual(block.instructions, [])
        self.assertEqual(block.terminator.op, "return")
        self.assertIsNone(block.terminator.value)

    def test_traversable_object_graph(self):
        ast = program(
            func("f", [("x", "int")], "int",
                 [vdecl("y", "int", intlit(2)),
                  assign("y", binop("*", var("x"), var("y"))),
                  return_(var("y"))])
        )
        fn = cir.lower_module(ast).functions[0]
        # Walk the graph like a future SSA pass would.
        ops = []
        for block in fn.blocks:
            for instr in block.instructions:
                ops.append(instr.op)
            ops.append(block.terminator.op)
        self.assertEqual(ops, ["const", "store",
                               "load", "load", "mul", "store",
                               "load", "return"])
        self.assertEqual([s.name for s in fn.slots], ["x", "y"])

    def test_function_order_matches_ast(self):
        ast = program(
            func("a", [], "void", []),
            func("b", [], "void", []),
            func("c", [], "void", []),
        )
        module = cir.lower_module(ast)
        self.assertEqual([f.name for f in module.functions], ["a", "b", "c"])

    def test_call_to_later_function(self):
        ast = program(
            func("first", [], "int", [return_(call("second", intlit(3)))]),
            func("second", [("x", "int")], "int", [return_(var("x"))]),
        )
        module = cir.lower_module(ast)
        ops = [i.op for i in module.functions[0].entry.instructions]
        self.assertIn("call", ops)


class ControlFlowTest(unittest.TestCase):
    def _blocks_by_terminator(self, fn):
        return [b.terminator.op for b in fn.blocks]

    def test_if_without_else_structure(self):
        ast = program(func(
            "f", [("c", "bool")], "void",
            [if_(var("c"), [vdecl("x", "int", intlit(1))])]))
        fn = cir.lower_module(ast).functions[0]
        # entry (cbr), then (br), join (return)
        terms = self._blocks_by_terminator(fn)
        self.assertEqual(terms, ["cbr", "br", "return"])
        entry = fn.entry
        self.assertEqual(entry.terminator.targets[1], fn.blocks[-1])

    def test_if_else_both_return_prunes_join(self):
        ast = program(func(
            "f", [("c", "bool")], "int",
            [if_(var("c"), [return_(intlit(1))],
                 [return_(intlit(2))])]))
        fn = cir.lower_module(ast).functions[0]
        labels = [b.label for b in fn.blocks]
        self.assertEqual(len(labels), 3)  # entry, then, else; no dead join
        # Every block must be terminated; the join must not linger.
        for block in fn.blocks:
            self.assertIsNotNone(block.terminator)

    def test_if_else_one_branch_falls_through_keeps_join(self):
        ast = program(func(
            "f", [("c", "bool")], "int",
            [vdecl("x", "int", intlit(0)),
             if_(var("c"), [assign("x", intlit(1))],
                 [return_(intlit(2))]),
             return_(var("x"))]))
        fn = cir.lower_module(ast).functions[0]
        self.assertGreaterEqual(len(fn.blocks), 4)

    def test_while_structure(self):
        ast = program(func(
            "f", [("n", "int")], "int",
            [vdecl("i", "int", intlit(0)),
             vdecl("total", "int", intlit(0)),
             while_(binop("<", var("i"), var("n")),
                    [assign("total", binop("+", var("total"), var("i"))),
                     assign("i", binop("+", var("i"), intlit(1)))]),
             return_(var("total"))]))
        fn = cir.lower_module(ast).functions[0]
        terminators = [b.terminator.op for b in fn.blocks]
        # entry br -> condition cbr, body br back, exit return
        self.assertEqual(terminators.count("cbr"), 1)
        self.assertEqual(terminators.count("br"), 2)
        self.assertEqual(terminators.count("return"), 1)
        # condition cbr targets: body then exit
        cond = next(b for b in fn.blocks
                    if b.terminator.op == "cbr")
        body, exit_block = cond.terminator.targets
        self.assertEqual(body.terminator.op, "br")
        self.assertIs(body.terminator.targets[0], cond)  # back edge
        self.assertEqual(exit_block.terminator.op, "return")

    def test_while_first_statement_uses_entry_as_condition(self):
        ast = program(func(
            "f", [("c", "bool")], "void",
            [while_(var("c"), [] )]))
        fn = cir.lower_module(ast).functions[0]
        # Entry itself evaluates the condition: cbr in block b0.
        self.assertEqual(fn.entry.terminator.op, "cbr")

    def test_statements_after_return_produce_no_ir(self):
        ast = program(func(
            "f", [], "int",
            [return_(intlit(1)),
             return_(intlit(2))]))
        fn = cir.lower_module(ast).functions[0]
        self.assertEqual(len(fn.blocks), 1)
        self.assertEqual(len(fn.entry.instructions), 1)
        self.assertEqual(fn.entry.instructions[0].op, "const")

    def test_nested_blocks_and_while(self):
        ast = program(func(
            "f", [("n", "int")], "int",
            [block([
                vdecl("x", "int", intlit(0)),
                while_(binop("<", var("x"), var("n")),
                       [if_(binop("==", var("x"), intlit(3)),
                            [assign("x", binop("+", var("x"), intlit(1)))])]),
             ]),
             # x is scoped inside the block above, so declare another
             vdecl("y", "int", intlit(7)),
             return_(var("y"))]))
        # Must lower without error and stay connected.
        fn = cir.lower_module(ast).functions[0]
        reachable = {id(b) for b in fn.blocks}
        worklist = [fn.entry]
        seen = set()
        while worklist:
            b = worklist.pop()
            if id(b) in seen:
                continue
            seen.add(id(b))
            worklist.extend(b.terminator.targets)
        self.assertEqual(seen, reachable)


class ShortCircuitTest(unittest.TestCase):
    def _lower_bool(self, expr):
        ast = program(func("f", [("a", "bool"), ("b", "bool")], "bool",
                           [return_(expr)]))
        return cir.lower_module(ast).functions[0]

    def test_and_branches_on_left(self):
        fn = self._lower_bool(binop("and", var("a"), var("b")))
        # entry cbr, rhs br, short br, join return => 4 blocks
        self.assertEqual(len(fn.blocks), 4)
        entry = fn.entry
        self.assertEqual(entry.terminator.op, "cbr")
        rhs_block, short_block = entry.terminator.targets
        # The right operand is evaluated in exactly one block: the true
        # successor of the entry branch.
        rhs_loads = [i for i in rhs_block.instructions if i.op == "load"]
        self.assertEqual(len(rhs_loads), 1)  # loads b
        # Short-circuit block materialises false for `and`.
        consts = [i for i in short_block.instructions if i.op == "const"]
        self.assertEqual(len(consts), 1)
        self.assertIs(consts[0].value, False)
        # Both paths merge in the same join block.
        self.assertIs(rhs_block.terminator.targets[0],
                      short_block.terminator.targets[0])

    def test_or_short_circuit_value_is_true(self):
        fn = self._lower_bool(binop("or", var("a"), var("b")))
        entry = fn.entry
        # or: true successor short-circuits, false successor evaluates RHS.
        short_block, rhs_block = entry.terminator.targets
        consts = [i for i in short_block.instructions if i.op == "const"]
        self.assertIs(consts[0].value, True)
        rhs_loads = [i for i in rhs_block.instructions if i.op == "load"]
        self.assertEqual(len(rhs_loads), 1)

    def test_right_operand_not_unconditionally_evaluated(self):
        # The RHS call's instructions must only appear on the branch that
        # needs it, never in the entry block.
        ast = program(
            func("f", [("a", "bool")], "bool",
                 [return_(binop("and", var("a"),
                                call("probe")))]),
            func("probe", [], "bool", [return_(boollit(True))]),
        )
        fn = cir.lower_module(ast).functions[0]
        entry_ops = [i.op for i in fn.entry.instructions]
        self.assertNotIn("call", entry_ops)
        rhs_block = fn.entry.terminator.targets[0]
        self.assertIn("call", [i.op for i in rhs_block.instructions])


class NameResolutionTest(unittest.TestCase):
    def test_visible_from_next_statement(self):
        ok = program(func("f", [], "int",
                          [vdecl("x", "int", intlit(1)),
                           return_(var("x"))]))
        cir.lower_module(ok)  # no error

    def test_reference_before_declaration(self):
        ast = program(func("f", [], "int",
                           [vdecl("x", "int", var("x")),
                            return_(intlit(0))]))
        with self.assertRaises(errors.UndefinedSymbolError):
            cir.lower_module(ast)

    def test_inner_shadows_outer(self):
        ast = program(func(
            "f", [("x", "int")], "int",
            [block([vdecl("x", "bool", boollit(True))]),
             return_(var("x"))]))
        fn = cir.lower_module(ast).functions[0]
        # Two slots: parameter x and the shadowing bool x.
        self.assertEqual(len(fn.slots), 2)
        self.assertEqual(fn.slots[0].type, "int")
        self.assertEqual(fn.slots[1].type, "bool")
        # The final load reads slot 0 (outer x), not the inner slot.
        load = fn.entry.instructions[-1]
        self.assertEqual(load.op, "load")
        self.assertEqual(load.slot.id, 0)

    def test_duplicate_in_same_scope(self):
        ast = program(func(
            "f", [], "void",
            [vdecl("x", "int"), vdecl("x", "int")]))
        with self.assertRaises(errors.DuplicateSymbolError) as ctx:
            cir.lower_module(ast)
        self.assertIn("body[1]", str(ctx.exception))

    def test_undeclared_reference(self):
        ast = program(func("f", [], "void",
                           [assign("y", intlit(1))]))
        with self.assertRaises(errors.UndefinedSymbolError):
            cir.lower_module(ast)

    def test_duplicate_function_and_parameter(self):
        with self.assertRaises(errors.DuplicateSymbolError):
            cir.lower_module(program(
                func("a", [], "void", []),
                func("a", [], "void", [])))
        with self.assertRaises(errors.DuplicateSymbolError):
            cir.lower_module(program(
                func("a", [("x", "int"), ("x", "bool")], "void", [])))

    def test_call_unknown_function(self):
        ast = program(func("f", [], "int",
                           [return_(call("ghost", intlit(1)))]))
        with self.assertRaises(errors.UndefinedSymbolError):
            cir.lower_module(ast)


class TypeCheckTest(unittest.TestCase):
    def check(self, ast):
        with self.assertRaises(errors.TypeCheckError):
            cir.lower_module(ast)

    def test_arithmetic_requires_int(self):
        self.check(program(func(
            "f", [("b", "bool")], "void",
            [vdecl("x", "int", binop("+", var("b"), intlit(1)))])))
        self.check(program(func(
            "f", [("b", "bool")], "void",
            [vdecl("x", "bool", binop("*", intlit(1), intlit(2)))])))

    def test_boolean_ops_require_bool(self):
        self.check(program(func(
            "f", [], "void",
            [vdecl("x", "bool",
                   binop("and", boollit(True), intlit(1)))])))

    def test_comparison_yields_bool(self):
        self.check(program(func(
            "f", [], "void",
            [vdecl("x", "int",
                   binop("<", intlit(1), intlit(2)))])))

    def test_comparison_requires_int_operands(self):
        self.check(program(func(
            "f", [("a", "bool"), ("b", "bool")], "void",
            [vdecl("x", "bool",
                   binop("==", var("a"), var("b")))])))

    def test_assignment_matches_declared_type(self):
        self.check(program(func(
            "f", [], "void",
            [vdecl("x", "int"),
             assign("x", boollit(True))])))

    def test_init_matches_declared_type(self):
        self.check(program(func(
            "f", [], "void",
            [vdecl("x", "bool", intlit(0))])))

    def test_return_type_mismatch(self):
        self.check(program(func("f", [], "int",
                                [return_(boollit(False))])))

    def test_void_rules(self):
        self.check(program(func("f", [], "void",
                                [return_(intlit(1))])))
        self.check(program(func("f", [], "int", [return_()])))

    def test_condition_must_be_bool(self):
        self.check(program(func(
            "f", [], "void",
            [if_(intlit(1), [])])))
        self.check(program(func(
            "f", [], "void",
            [while_(intlit(0), [])])))

    def test_call_arity_and_arg_types(self):
        callee = func("g", [("x", "int")], "int", [return_(var("x"))])
        self.check(program(
            func("f", [], "int", [return_(call("g"))]), callee))
        self.check(program(
            func("f", [], "int",
                 [return_(call("g", boollit(True)))]), callee))

    def test_void_call_is_not_a_value(self):
        callee = func("g", [], "void", [])
        self.check(program(
            func("f", [], "int",
                 [vdecl("x", "int", call("g")),
                  return_(intlit(0))]), callee))


class ReturnPathTest(unittest.TestCase):
    def test_missing_return_reported(self):
        with self.assertRaises(errors.MissingReturnError):
            cir.lower_module(program(func("f", [], "int", [])))

    def test_if_only_then_returns_is_not_complete(self):
        ast = program(func(
            "f", [("c", "bool")], "int",
            [if_(var("c"), [return_(intlit(1))])]))
        with self.assertRaises(errors.MissingReturnError):
            cir.lower_module(ast)

    def test_if_else_both_return_is_complete(self):
        ast = program(func(
            "f", [("c", "bool")], "int",
            [if_(var("c"), [return_(intlit(1))],
                 [return_(intlit(2))])]))
        cir.lower_module(ast)  # no error

    def test_while_is_not_assumed_to_terminate(self):
        ast = program(func(
            "f", [("c", "bool")], "int",
            [while_(var("c"), [return_(intlit(1))])]))
        with self.assertRaises(errors.MissingReturnError):
            cir.lower_module(ast)

    def test_void_function_needs_no_return(self):
        cir.lower_module(program(func(
            "f", [("x", "int")], "void",
            [vdecl("y", "int", var("x")),
             while_(binop("<", var("y"), intlit(3)),
                    [assign("y", binop("+", var("y"), intlit(1)))])])))


class InvalidAstTest(unittest.TestCase):
    def assertInvalid(self, ast, fragment=""):
        with self.assertRaises(errors.InvalidAstError) as ctx:
            cir.lower_module(ast)
        if fragment:
            self.assertIn(fragment, str(ctx.exception))

    def test_not_a_dict(self):
        self.assertInvalid([], "$")
        self.assertInvalid("nope", "$")

    def test_missing_required_fields(self):
        self.assertInvalid({}, "$.functions")
        self.assertInvalid(
            {"functions": [{"kind": "function", "name": "f"}]},
            "$.functions[0].params")

    def test_unknown_node_kinds(self):
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "void",
                 "body": [{"kind": "goto"}]}]},
            "$.functions[0].body[0].kind")
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "void",
                 "body": [vdecl("x", "int", {"kind": "wat"})]}]},
            "init.kind")

    def test_illegal_type_names(self):
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "string", "body": []}]},
            "return_type")
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "void",
                 "body": [vdecl("x", "float")]}]},
            "body[0].type")

    def test_structural_path_for_bad_expression(self):
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "int",
                 "body": [return_(binop("-", intlit(1),
                                        {"kind": "int"}))]}]},
            "$.functions[0].body[0].value.right.value")

    def test_bool_is_not_int_literal(self):
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "void",
                 "body": [vdecl("x", "int", intlit(True))]}]},
            "body[0].init.value")

    def test_unknown_operator(self):
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "void",
                 "body": [vdecl("x", "int",
                                binop("%", intlit(1), intlit(2)))]}]},
            "op")

    def test_extra_field_rejected(self):
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": [],
                 "return_type": "void", "body": [],
                 "mystery": 1}]},
            "mystery")

    def test_bad_list_member_type(self):
        self.assertInvalid(
            {"functions": [
                {"kind": "function", "name": "f", "params": ["x"],
                 "return_type": "void", "body": []}]},
            "$.functions[0].params[0]")


if __name__ == "__main__":
    unittest.main()
