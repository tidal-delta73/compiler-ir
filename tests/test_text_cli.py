"""Tests for deterministic text rendering and the public API."""

import subprocess
import sys
import unittest
from pathlib import Path

import compiler_ir as cir

REPO_ROOT = Path(__file__).resolve().parents[1]


def intlit(v):
    return {"kind": "int", "value": v}


def boollit(v):
    return {"kind": "bool", "value": v}


def var(n):
    return {"kind": "var", "name": n}


def binop(op, l, r):
    return {"kind": "binary", "op": op, "left": l, "right": r}


def call(n, *a):
    return {"kind": "call", "name": n, "args": list(a)}


def vdecl(n, t, init=None):
    node = {"kind": "var_decl", "name": n, "type": t}
    if init is not None:
        node["init"] = init
    return node


def assign(n, v):
    return {"kind": "assign", "target": n, "value": v}


def return_(v=None):
    return {"kind": "return", "value": v} if v is not None \
        else {"kind": "return"}


def if_(c, t, e=None):
    node = {"kind": "if", "cond": c, "then": t}
    if e is not None:
        node["else"] = e
    return node


def while_(c, body):
    return {"kind": "while", "cond": c, "body": body}


def func(name, params, ret, body):
    return {"kind": "function", "name": name,
            "params": [{"name": p, "type": t} for p, t in params],
            "return_type": ret, "body": body}


def program(*fns):
    return {"functions": list(fns)}


class RenderTest(unittest.TestCase):
    def test_void_function_text(self):
        text = cir.render_module(
            cir.lower_module(program(func("noop", [], "void", []))))
        self.assertEqual(text,
                         "func @noop() -> void {\n"
                         "b0:\n"
                         "  return\n"
                         "}\n")

    def test_signature_and_instructions(self):
        ast = program(func(
            "add", [("x", "int"), ("y", "int")], "int",
            [return_(binop("+", var("x"), var("y")))]))
        text = cir.render_module(cir.lower_module(ast))
        expected = (
            "func @add(x: int, y: int) -> int {\n"
            "  slot %s0: int ; x\n"
            "  slot %s1: int ; y\n"
            "b0:\n"
            "  %t0: int = load %s0\n"
            "  %t1: int = load %s1\n"
            "  %t2: int = add %t0 %t1\n"
            "  return %t2: int\n"
            "}\n"
        )
        self.assertEqual(text, expected)

    def test_bool_literal_rendering(self):
        ast = program(func("f", [], "bool",
                           [return_(binop("or", boollit(False),
                                          boollit(True)))]))
        text = cir.render_module(cir.lower_module(ast))
        self.assertIn("const false", text)
        self.assertIn("const true", text)
        self.assertIn("cbr %t0", text)
        self.assertEqual(text.count("\n  br "), 2)

    def test_if_else_text(self):
        ast = program(func(
            "choose", [("c", "bool")], "int",
            [if_(var("c"), [return_(intlit(1))],
                 [return_(intlit(2))])]))
        text = cir.render_module(cir.lower_module(ast))
        self.assertIn("func @choose(c: bool) -> int {", text)
        self.assertIn("cbr %t0 b1 b2", text)
        self.assertNotIn("b3", text)  # unreachable join removed

    def test_while_text_uses_stable_numbering(self):
        ast = program(func(
            "loop", [("n", "int")], "int",
            [vdecl("i", "int", intlit(0)),
             while_(binop("<", var("i"), var("n")),
                    [assign("i", binop("+", var("i"), intlit(1)))]),
             return_(var("i"))]))
        text = cir.render_module(cir.lower_module(ast))
        lines = text.splitlines()
        # Block labels appear in ascending order: entry, condition,
        # body, exit.
        labels = [line for line in lines
                  if line.startswith("b") and line.endswith(":")]
        self.assertEqual(labels, ["b0:", "b1:", "b2:", "b3:"])
        # The body back-edge names the condition block.
        self.assertIn("br b1", text)

    def test_multiple_functions_separated_by_blank_line(self):
        ast = program(
            func("a", [], "void", []),
            func("b", [], "void", []),
        )
        text = cir.render_module(cir.lower_module(ast))
        self.assertEqual(text.count("}\n"), 2)
        self.assertIn("}\n\nfunc @b", text)

    def test_numbering_is_depth_first_and_dense(self):
        ast = program(func(
            "f", [("a", "bool"), ("b", "bool")], "bool",
            [if_(var("a"),
                 [return_(binop("and", var("a"), var("b")))]),
             return_(boollit(False))]))
        fn = cir.lower_module(ast).functions[0]
        self.assertEqual([b.id for b in fn.blocks],
                         list(range(len(fn.blocks))))
        self.assertEqual([t.id for t in fn.temps],
                         list(range(len(fn.temps))))

    def test_text_does_not_contain_object_addresses(self):
        ast = program(func("f", [("x", "int")], "int",
                           [return_(binop("+", var("x"), intlit(1)))]))
        text = cir.render_module(cir.lower_module(ast))
        self.assertNotIn("0x", text)
        self.assertNotIn(" object at ", text)


class DeterminismTest(unittest.TestCase):
    SAMPLE = program(
        func("classify", [("n", "int")], "int",
             [vdecl("flag", "bool",
                    binop(">", var("n"), intlit(0))),
              if_(binop("and", var("flag"),
                        binop("<", var("n"), intlit(10))),
                  [return_(intlit(1))],
                  [while_(binop("==", var("n"), intlit(10)),
                          [assign("n", binop("-", var("n"), intlit(1)))]),
                   return_(intlit(0))])]),
        func("helper", [("a", "bool"), ("b", "bool")], "bool",
             [return_(binop("or", var("a"), var("b")))]),
    )

    def test_repeated_renders_are_identical(self):
        module_a = cir.lower_module(self.SAMPLE)
        module_b = cir.lower_module(self.SAMPLE)
        self.assertEqual(cir.render_module(module_a),
                         cir.render_module(module_b))

    def test_identical_bytes_in_separate_processes(self):
        import json
        import os

        driver = (
            "import sys; sys.path.insert(0, %r);"
            "import json, compiler_ir as cir;"
            "ast = json.loads(sys.stdin.read());"
            "sys.stdout.write(cir.render_module(cir.lower_module(ast)))"
            % str(REPO_ROOT)
        )
        payload = json.dumps(self.SAMPLE)
        local = cir.render_module(cir.lower_module(self.SAMPLE))

        # PYTHONHASHSEED changes dict/set hashing between processes; the
        # rendered bytes must not depend on hash order, object addresses
        # or collection iteration order.
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run(
                [sys.executable, "-c", driver], input=payload,
                capture_output=True, text=True, check=True,
                cwd=REPO_ROOT, env=env)
            self.assertEqual(
                result.stdout, local,
                msg=f"render differs under PYTHONHASHSEED={seed}")


class CliRegressionTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "compiler_ir", *args],
            cwd=REPO_ROOT, capture_output=True, text=True)

    def test_version(self):
        result = self.run_cli("version")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), cir.__version__)

    def test_help(self):
        result = self.run_cli("help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("commands:", result.stdout)

    def test_no_args_prints_help(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)

    def test_unknown_command_exit_code(self):
        result = self.run_cli("compile", "something")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command: compile", result.stderr)

    def test_no_compile_command_added(self):
        result = self.run_cli("compile")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
