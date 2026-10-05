"""Tests for AST validation, semantic analysis, lowering and text IR."""
import unittest

from compiler_ir import (
    DuplicateSymbolError,
    InvalidAstError,
    MissingReturnError,
    TypeCheckError,
    UndefinedSymbolError,
    emit_ir,
    lower_module,
    render_module,
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


# Two mutually recursive-ish int functions used across tests.
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


class LoweringTests(unittest.TestCase):
    def test_empty_void_function(self):
        text = emit_ir(program(func("noop", [], "void", [])))
        self.assertIn("function noop() -> void", text)
        self.assertIn("b0:", text)
        # The entry block's terminator is an explicit bare return.
        body = text[text.index("b0:"):]
        self.assertIn("    return\n", body)

    def test_nonvoid_with_constant_return(self):
        text = emit_ir(program(func("answer", [], "int", [ret(int_(42))])))
        self.assertIn("const 42", text)
        self.assertIn("return %t0", text)

    def test_parameters_occupy_first_slots(self):
        module = lower_module(
            program(func("add", [param("a", "int"), param("b", "int")], "int",
                        [ret(arith("add", var("a"), var("b")))]))
        )
        fn = module.functions[0]
        self.assertEqual([p.slot.id for p in fn.params], [0, 1])
        self.assertEqual(fn.locals, [])

    def test_function_call_arguments_and_result(self):
        text = emit_ir(
            program(
                func("id", [param("x", "bool")], "bool", [ret(var("x"))]),
                func("use", [], "bool", [ret(call("id", [bool_(True)]))]),
            )
        )
        self.assertIn("call id(", text)

    def test_forward_function_call_resolves(self):
        # callee is declared after the caller; the function table is global.
        text = emit_ir(
            program(
                func("use", [], "int", [ret(call("later", [int_(5)]))]),
                func("later", [param("x", "int")], "int", [ret(var("x"))]),
            )
        )
        self.assertIn("call later(%t0)", text)

    def test_short_circuit_and_branches(self):
        # and: right operand evaluated only on the true path.
        text = emit_ir(
            program(
                func(
                    "f", [], "bool",
                    [
                        let(
                            "r", "bool",
                            logical("and", bool_(True),
                                    compare("eq", int_(1), int_(1))),
                        ),
                        ret(var("r")),
                    ],
                )
            )
        )
        # The compare must live in the branch reached when left is true.
        lines = text.splitlines()
        br_line = next(i for i, l in enumerate(lines) if l.strip().startswith("br %t0"))
        compare_line = next(i for i, l in enumerate(lines) if "compare eq" in l)
        self.assertGreater(compare_line, br_line)
        # The false (short-circuit) path must contain const false, no compare.
        self.assertIn("const false", text)

    def test_short_circuit_or_does_not_eval_right_when_true(self):
        module = lower_module(
            program(
                func(
                    "f", [], "bool",
                    [
                        let(
                            "r", "bool",
                            logical("or", bool_(True),
                                    compare("lt", int_(1), int_(2))),
                        ),
                        ret(var("r")),
                    ],
                )
            )
        )
        fn = module.functions[0]
        # b0 branches on the left literal; the true branch only materializes
        # the constant true result and must contain no compare instruction.
        true_block = fn.blocks[1]
        self.assertTrue(
            all("compare" not in i.op for i in true_block.instructions)
        )

    def test_if_merges_only_when_reachable(self):
        # Both branches return -> no merge block; exactly three blocks
        # (entry, true, false).
        module = lower_module(
            program(
                func(
                    "f", [param("c", "bool")], "int",
                    [if_(var("c"), [ret(int_(1))], [ret(int_(2))])],
                )
            )
        )
        self.assertEqual(len(module.functions[0].blocks), 3)

        # Only the else returns -> a merge is required (4 blocks), and code
        # after the if is reachable.
        module2 = lower_module(
            program(
                func(
                    "g", [param("c", "bool")], "int",
                    [if_(var("c"), [ret(int_(1))], []), ret(int_(2))],
                )
            )
        )
        self.assertEqual(len(module2.functions[0].blocks), 4)

    def test_no_unreachable_empty_merge_after_terminal_if(self):
        text = emit_ir(
            program(
                func(
                    "f", [param("c", "bool")], "int",
                    [if_(var("c"), [ret(int_(1))], [ret(int_(2))])],
                )
            )
        )
        # No jump to an immediately-returning empty block: the function ends
        # with the two branch return paths only.
        self.assertNotIn("jump b3", text)

    def test_statements_after_return_emit_nothing(self):
        module = lower_module(
            program(
                func(
                    "f", [], "int",
                    [ret(int_(1)), ret(int_(2))],
                )
            )
        )
        fn = module.functions[0]
        # Only the first return's const survives; b0 has a single terminator.
        self.assertEqual(len(fn.blocks), 1)
        self.assertEqual(len(fn.blocks[0].instructions), 1)
        consts = [i for i in fn.blocks[0].instructions if i.op == "const"]
        self.assertEqual(consts[0].value, 1)

    def test_shadowing_is_allowed(self):
        text = emit_ir(
            program(
                func(
                    "f", [], "int",
                    [
                        let("x", "int", int_(1)),
                        block([let("x", "int", int_(2)), ret(var("x"))]),
                        ret(int_(0)),
                    ],
                )
            )
        )
        self.assertIn("const 2", text)

    def test_let_not_visible_in_own_initializer(self):
        # The reference to x inside its own initializer is unbound.
        with self.assertRaises(UndefinedSymbolError):
            lower_module(
                program(func("f", [], "int",
                             [let("x", "int", var("x")), ret(var("x"))]))
            )

    def test_visible_from_following_statement(self):
        # Reference in the *next* statement is fine.
        text = emit_ir(
            program(
                func("f", [], "int",
                     [let("x", "int", int_(1)), ret(var("x"))])
            )
        )
        self.assertIn("return %v0", text)

    def test_block_numbering_is_dfs(self):
        module = lower_module(program(COUNTER))
        ids = [b.id for b in module.functions[0].blocks]
        self.assertEqual(ids, list(range(len(ids))))

    def test_determinism_byte_identical(self):
        ast = program(COUNTER)
        first = emit_ir(ast)
        second = emit_ir(ast)
        self.assertEqual(first, second)
        # Re-ordering the (single) function table source dict keys must not
        # matter; rebuild with reversed insertion to stress key independence.
        rebuilt = {
            "functions": [
                {
                    "body": COUNTER["body"],
                    "ret_type": COUNTER["ret_type"],
                    "params": COUNTER["params"],
                    "name": COUNTER["name"],
                }
            ]
        }
        self.assertEqual(first, emit_ir(rebuilt))


class ValidationTests(unittest.TestCase):
    def assertPath(self, ctx, suffix):
        self.assertIn(suffix, ctx.exception.path)

    def test_not_a_dict(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module([])
        self.assertPath(ctx, "module")

    def test_missing_functions(self):
        with self.assertRaises(InvalidAstError):
            lower_module({})

    def test_missing_function_field(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module({"functions": [{"name": "f"}]})
        self.assertIn("params", str(ctx.exception))
        self.assertPath(ctx, "module.functions[0]")

    def test_illegal_type_name(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module(
                program(func("f", [], "string", [])))
        self.assertPath(ctx, "ret_type")

    def test_unknown_statement_kind(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module(
                program(func("f", [], "void", [{"kind": "print"}])))
        self.assertPath(ctx, "body[0]")

    def test_unknown_expression_kind(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module(
                program(func("f", [], "int",
                             [ret({"kind": "string", "value": "x"})])))
        self.assertPath(ctx, "body[0].value")

    def test_unknown_operator(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module(
                program(func("f", [], "int",
                             [ret(arith("pow", int_(1), int_(2)))])))
        self.assertPath(ctx, "op")

    def test_wrong_literal_python_type(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module(
                program(func("f", [], "int", [ret(int_("1"))])))
        self.assertPath(ctx, "value")

    def test_bool_literal_rejects_int(self):
        with self.assertRaises(InvalidAstError):
            lower_module(
                program(func("f", [], "bool",
                             [ret({"kind": "bool", "value": 1})])))

    def test_extra_field_rejected(self):
        bad = func("f", [], "void", [])
        bad["extra"] = 1
        with self.assertRaises(InvalidAstError):
            lower_module(program(bad))

    def test_deep_path_in_nested_if(self):
        with self.assertRaises(InvalidAstError) as ctx:
            lower_module(
                program(
                    func(
                        "f", [], "void",
                        [if_(bool_(True),
                             [if_(bool_(True), [{"kind": "nope"}], [])])],
                    )
                )
            )
        self.assertIn("then[0].then[0]", ctx.exception.path)


class SemanticTests(unittest.TestCase):
    def test_duplicate_variable_same_scope(self):
        with self.assertRaises(DuplicateSymbolError):
            lower_module(
                program(
                    func(
                        "f", [], "void",
                        [let("x", "int", int_(1)),
                         let("x", "int", int_(2))],
                    )
                )
            )

    def test_shadowing_not_duplicate(self):
        lower_module(
            program(
                func(
                    "f", [], "void",
                    [
                        let("x", "int", int_(1)),
                        block([let("x", "bool", bool_(True))]),
                    ],
                )
            )
        )

    def test_duplicate_parameter(self):
        with self.assertRaises(DuplicateSymbolError):
            lower_module(
                program(func("f", [param("x", "int"), param("x", "int")],
                             "void", [])))

    def test_duplicate_function(self):
        with self.assertRaises(DuplicateSymbolError):
            lower_module(
                program(func("f", [], "void", []), func("f", [], "void", [])))

    def test_undefined_variable(self):
        with self.assertRaises(UndefinedSymbolError):
            lower_module(
                program(func("f", [], "int", [ret(var("missing"))])))

    def test_assignment_to_undefined(self):
        with self.assertRaises(UndefinedSymbolError):
            lower_module(
                program(func("f", [], "void",
                             [assign("missing", int_(1))])))

    def test_undefined_function_call(self):
        with self.assertRaises(UndefinedSymbolError):
            lower_module(
                program(func("f", [], "int",
                             [ret(call("ghost", []))])))

    def test_assign_type_mismatch(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(
                    func("f", [], "void",
                         [let("x", "int", int_(1)),
                          assign("x", bool_(True))])
                )
            )

    def test_let_type_mismatch(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(func("f", [], "void",
                             [let("x", "int", bool_(False))])))

    def test_arith_requires_int(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(func("f", [], "int",
                             [ret(arith("add", bool_(True), int_(1)))])))

    def test_logical_requires_bool(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(func("f", [], "bool",
                             [ret(logical("and", int_(1), bool_(True)))])))

    def test_compare_mismatched_operands(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(func("f", [], "bool",
                             [ret(compare("eq", int_(1), bool_(True)))])))

    def test_compare_result_is_bool(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(func("f", [], "int",
                             [ret(compare("eq", int_(1), int_(1)))])))

    def test_if_condition_must_be_bool(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(func("f", [], "void",
                             [if_(int_(1), [], [])])))

    def test_return_type_mismatch(self):
        with self.assertRaises(TypeCheckError):
            lower_module(program(func("f", [], "int", [ret(bool_(True))])))

    def test_void_return_with_value(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(func("f", [], "void", [ret(int_(1))])))

    def test_nonvoid_empty_return(self):
        with self.assertRaises(TypeCheckError):
            lower_module(program(func("f", [], "int", [ret(None)])))

    def test_call_arity_mismatch(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(
                    func("id", [param("x", "int")], "int", [ret(var("x"))]),
                    func("g", [], "int", [ret(call("id", []))]),
                )
            )

    def test_call_argument_type_mismatch(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(
                    func("id", [param("x", "int")], "int", [ret(var("x"))]),
                    func("g", [], "int",
                         [ret(call("id", [bool_(True)]))]),
                )
            )

    def test_call_to_void_as_value(self):
        with self.assertRaises(TypeCheckError):
            lower_module(
                program(
                    func("p", [], "void", []),
                    func("g", [], "int", [ret(call("p", []))]),
                )
            )


class ReturnReachabilityTests(unittest.TestCase):
    def test_missing_return_fallthrough(self):
        with self.assertRaises(MissingReturnError):
            lower_module(program(func("f", [], "int", [])))

    def test_missing_return_after_while(self):
        # while may run zero times; code after it is reachable.
        with self.assertRaises(MissingReturnError):
            lower_module(
                program(
                    func(
                        "f", [], "int",
                        [while_(bool_(True), [ret(int_(1))])],
                    )
                )
            )

    def test_return_after_while_is_fine(self):
        lower_module(
            program(
                func(
                    "f", [], "int",
                    [while_(bool_(True), []), ret(int_(0))],
                )
            )
        )

    def test_if_must_return_on_both_branches(self):
        with self.assertRaises(MissingReturnError):
            lower_module(
                program(
                    func("f", [param("c", "bool")], "int",
                         [if_(var("c"), [ret(int_(1))], [])])
                )
            )

    def test_both_branches_return_satisfies(self):
        lower_module(
            program(
                func(
                    "f", [param("c", "bool")], "int",
                    [if_(var("c"), [ret(int_(1))], [ret(int_(2))])],
                )
            )
        )

    def test_nested_blocks_with_terminal_return(self):
        # if/else where the then-block ends in a nested block that returns.
        lower_module(
            program(
                func(
                    "f", [param("c", "bool")], "int",
                    [
                        if_(
                            var("c"),
                            [block([ret(int_(1))])],
                            [ret(int_(2))],
                        )
                    ],
                )
            )
        )

    def test_void_function_needs_no_return(self):
        lower_module(
            program(
                func(
                    "f", [], "void",
                    [let("x", "int", int_(1)),
                     if_(bool_(True), [assign_dummy()], [])],
                )
            )
        )


def assign_dummy():
    # Helper kept module-level so the void test can mutate a declared local.
    return assign("x", int_(2))


class CliTests(unittest.TestCase):
    def test_module_version_attribute(self):
        import compiler_ir

        self.assertEqual(compiler_ir.__version__, "0.1.0")

    def test_main_version_and_help(self):
        from compiler_ir.__main__ import main

        self.assertEqual(main(["version"]), 0)
        self.assertEqual(main(["help"]), 0)
        self.assertEqual(main([]), 0)

    def test_unknown_command_exit_code(self):
        from compiler_ir.__main__ import main

        self.assertEqual(main(["compile"]), 2)


if __name__ == "__main__":
    unittest.main()
