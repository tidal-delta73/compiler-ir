"""Name resolution, type checking and return-path analysis.

Runs over the structurally validated AST before any IR is produced, so a
semantic error never yields a partial :class:`~compiler_ir.ir.Module`.
"""

from .ast_schema import (
    ARITHMETIC_OPS,
    BOOLEAN_OPS,
    COMPARISON_OPS,
)
from .errors import (
    DuplicateSymbolError,
    InvalidAstError,
    MissingReturnError,
    TypeCheckError,
    UndefinedSymbolError,
)


class _Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.names = {}

    def declare(self, name, type_name, path):
        if name in self.names:
            raise DuplicateSymbolError(
                f"{path}: {name!r} is already declared in this scope")
        self.names[name] = type_name

    def resolve(self, name):
        scope = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None


class _Checker:
    def __init__(self):
        # name -> (return_type, [parameter types])
        self.functions = {}
        self.current_return = None
        self.path = ""

    # -- programs -------------------------------------------------------

    def check_program(self, root):
        path = "$"
        functions = root["functions"]
        # Function names are registered first, in AST order, so calls may
        # refer to functions declared later.
        for i, fn in enumerate(functions):
            fpath = f"{path}.functions[{i}]"
            name = fn["name"]
            if name in self.functions:
                raise DuplicateSymbolError(
                    f"{fpath}: function {name!r} is already defined")
            param_types = [p["type"] for p in fn["params"]]
            self.functions[name] = (fn["return_type"], param_types, fpath)
        for i, fn in enumerate(functions):
            self._check_function(fn, f"{path}.functions[{i}]")

    def _check_function(self, fn, path):
        self.current_return = fn["return_type"]
        scope = _Scope()
        seen_params = set()
        for i, param in enumerate(fn["params"]):
            if param["name"] in seen_params:
                raise DuplicateSymbolError(
                    f"{path}.params[{i}]: parameter {param['name']!r} "
                    "is declared more than once")
            seen_params.add(param["name"])
            scope.declare(param["name"], param["type"],
                          f"{path}.params[{i}]")
        always = self._check_stmts(fn["body"], scope, f"{path}.body")
        if fn["return_type"] != "void" and not always:
            raise MissingReturnError(
                f"{path}: function {fn['name']!r} declares return type "
                f"{fn['return_type']!r} but has a reachable path "
                "without a return statement")

    # -- statements -----------------------------------------------------

    def _check_stmts(self, stmts, scope, path):
        """Returns True iff execution always returns within this list.

        Every statement is visited (so errors are reported everywhere),
        but once a statement guarantees a return the rest is unreachable.
        """
        returned = False
        for i, stmt in enumerate(stmts):
            spath = f"{path}[{i}]"
            always = self._check_stmt(stmt, scope, spath)
            if always:
                returned = True
        return returned

    def _check_stmt(self, stmt, scope, path):
        kind = stmt["kind"]
        if kind == "var_decl":
            if "init" in stmt:
                init_type = self._check_expr(stmt["init"], scope,
                                             f"{path}.init")
                if init_type != stmt["type"]:
                    raise TypeCheckError(
                        f"{path}.init: cannot initialise variable "
                        f"{stmt['name']!r} of type {stmt['type']!r} "
                        f"with a value of type {init_type!r}")
            # Visible from the next statement on; the initialiser itself
            # does not see the new name.
            scope.declare(stmt["name"], stmt["type"], path)
            return False
        if kind == "assign":
            value_type = self._check_expr(stmt["value"], scope,
                                          f"{path}.value")
            target_type = scope.resolve(stmt["target"])
            if target_type is None:
                raise UndefinedSymbolError(
                    f"{path}.target: assignment to undeclared variable "
                    f"{stmt['target']!r}")
            if value_type != target_type:
                raise TypeCheckError(
                    f"{path}: cannot assign a value of type {value_type!r} "
                    f"to variable {stmt['target']!r} of type "
                    f"{target_type!r}")
            return False
        if kind == "if":
            cond_type = self._check_expr(stmt["cond"], scope,
                                         f"{path}.cond")
            if cond_type != "bool":
                raise TypeCheckError(
                    f"{path}.cond: if condition must be bool, "
                    f"got {cond_type!r}")
            then_always = self._check_stmts(
                stmt["then"], _Scope(scope), f"{path}.then")
            if "else" in stmt:
                else_always = self._check_stmts(
                    stmt["else"], _Scope(scope), f"{path}.else")
                return then_always and else_always
            return False
        if kind == "while":
            cond_type = self._check_expr(stmt["cond"], scope,
                                         f"{path}.cond")
            if cond_type != "bool":
                raise TypeCheckError(
                    f"{path}.cond: while condition must be bool, "
                    f"got {cond_type!r}")
            self._check_stmts(stmt["body"], _Scope(scope), f"{path}.body")
            # A loop is not assumed to always execute or terminate.
            return False
        if kind == "return":
            if self.current_return == "void":
                if "value" in stmt and stmt["value"] is not None:
                    value_type = self._check_expr(
                        stmt["value"], scope, f"{path}.value")
                    raise TypeCheckError(
                        f"{path}.value: void function cannot return "
                        f"a value of type {value_type!r}")
            else:
                if "value" not in stmt or stmt["value"] is None:
                    raise TypeCheckError(
                        f"{path}: function must return a value of type "
                        f"{self.current_return!r}")
                value_type = self._check_expr(
                    stmt["value"], scope, f"{path}.value")
                if value_type != self.current_return:
                    raise TypeCheckError(
                        f"{path}.value: returned value has type "
                        f"{value_type!r}, expected "
                        f"{self.current_return!r}")
            return True
        if kind == "block":
            return self._check_stmts(stmt["body"], _Scope(scope),
                                    f"{path}.body")
        raise InvalidAstError(f"{path}: unknown statement kind {kind!r}")

    # -- expressions ----------------------------------------------------

    def _check_expr(self, expr, scope, path):
        kind = expr["kind"]
        if kind == "int":
            return "int"
        if kind == "bool":
            return "bool"
        if kind == "var":
            type_name = scope.resolve(expr["name"])
            if type_name is None:
                raise UndefinedSymbolError(
                    f"{path}: reference to undeclared variable "
                    f"{expr['name']!r}")
            return type_name
        if kind == "binary":
            op = expr["op"]
            left = self._check_expr(expr["left"], scope, f"{path}.left")
            right = self._check_expr(expr["right"], scope, f"{path}.right")
            if op in ARITHMETIC_OPS:
                if left != "int" or right != "int":
                    raise TypeCheckError(
                        f"{path}: operator {op!r} requires int operands, "
                        f"got {left!r} and {right!r}")
                return "int"
            if op in COMPARISON_OPS:
                if left != "int" or right != "int":
                    raise TypeCheckError(
                        f"{path}: operator {op!r} requires int operands, "
                        f"got {left!r} and {right!r}")
                return "bool"
            if op in BOOLEAN_OPS:
                if left != "bool" or right != "bool":
                    raise TypeCheckError(
                        f"{path}: operator {op!r} requires bool operands, "
                        f"got {left!r} and {right!r}")
                return "bool"
            raise InvalidAstError(f"{path}.op: unknown operator {op!r}")
        if kind == "call":
            signature = self.functions.get(expr["name"])
            if signature is None:
                raise UndefinedSymbolError(
                    f"{path}: call to undeclared function "
                    f"{expr['name']!r}")
            return_type, param_types, _ = signature
            args = expr["args"]
            if len(args) != len(param_types):
                raise TypeCheckError(
                    f"{path}: function {expr['name']!r} expects "
                    f"{len(param_types)} argument(s), got {len(args)}")
            for i, arg in enumerate(args):
                arg_type = self._check_expr(arg, scope,
                                            f"{path}.args[{i}]")
                if arg_type != param_types[i]:
                    raise TypeCheckError(
                        f"{path}.args[{i}]: argument {i} of "
                        f"{expr['name']!r} has type {arg_type!r}, "
                        f"expected {param_types[i]!r}")
            if return_type == "void":
                raise TypeCheckError(
                    f"{path}: void function {expr['name']!r} cannot be "
                    "used as a value")
            return return_type
        raise InvalidAstError(f"{path}: unknown expression kind {kind!r}")


def check_program(root):
    """Check a structurally valid program AST."""
    _Checker().check_program(root)
