"""Name resolution and type checking for the AST subset.

Visibility rules
----------------
* A ``let`` binds from the statement *after* its declaration statement.
  (The initializer therefore cannot reference the variable itself.)
* Nested blocks push a fresh scope; an inner ``let`` shadows an outer
  binding of the same name (and is *not* a duplicate).
* A second ``let`` for the same name in the *same* scope raises
  :class:`DuplicateSymbolError`; a reference to an unbound name raises
  :class:`UndefinedSymbolError`.

Type rules
----------
* Arithmetic accepts only ``int`` operands and yields ``int``.
* Boolean ``and``/``or`` accept only ``bool`` and yield ``bool``.
* All six comparisons accept matching operand types and yield ``bool``;
  ``void`` never appears in an expression.
* Assignment, ``return`` and call arguments must match declarations.
* Calls must resolve to a known function with matching arity and argument
  types; the call type is the callee return type, so calls to ``void``
  functions are statement-less and rejectable.

This pass also computes reachability: a reachable path leaving a non-void
function without a ``return`` raises :class:`MissingReturnError`.
"""
from typing import Optional

from .errors import (
    DuplicateSymbolError,
    MissingReturnError,
    TypeCheckError,
    UndefinedSymbolError,
)

INT = "int"
BOOL = "bool"
VOID = "void"


class _Func:
    __slots__ = ("name", "param_types", "ret_type")

    def __init__(self, name: str, param_types: list[str], ret_type: str):
        self.name = name
        self.param_types = param_types
        self.ret_type = ret_type


class _Scope:
    """One block scope; ``declared`` is ordered for deterministic errors."""

    __slots__ = ("parent", "declared")

    def __init__(self, parent: Optional["_Scope"]):
        self.parent = parent
        self.declared: dict[str, str] = {}

    def declare(self, name: str, typ: str) -> None:
        if name in self.declared:
            raise DuplicateSymbolError(
                f"duplicate declaration of {name!r} in the same scope"
            )
        self.declared[name] = typ

    def lookup(self, name: str) -> Optional[str]:
        scope: Optional[_Scope] = self
        while scope is not None:
            if name in scope.declared:
                return scope.declared[name]
            scope = scope.parent
        return None


class SemanticAnalyzer:
    """Check one validated program AST."""

    def __init__(self, ast: dict):
        self.ast = ast
        # Function table is built in AST order for deterministic messages.
        self.functions: dict[str, _Func] = {}
        self._build_function_table()

    def _build_function_table(self) -> None:
        for func in self.ast["functions"]:
            name = func["name"]
            if name in self.functions:
                raise DuplicateSymbolError(
                    f"duplicate function declaration {name!r}"
                )
            self.functions[name] = _Func(
                name, [p["type"] for p in func["params"]], func["ret_type"]
            )

    def check(self) -> None:
        for func in self.ast["functions"]:
            self._check_function(func)

    # -- functions ---------------------------------------------------------

    def _check_function(self, func: dict) -> None:
        scope = _Scope(None)
        for param in func["params"]:
            # Parameter names occupy the top-level function scope; duplicates
            # here (e.g. two params of the same name) are also duplicates.
            scope.declare(param["name"], param["type"])
        self._ret_type = func["ret_type"]
        self._func_name = func["name"]
        completed = self._check_stmts(func["body"], scope)
        if self._ret_type != VOID and not completed:
            raise MissingReturnError(
                f"function {func['name']!r}: missing return on a reachable path"
            )

    # -- statements --------------------------------------------------------

    def _check_stmts(self, stmts: list, scope: _Scope) -> bool:
        """Return True if every path through these statements returns."""
        current = scope
        for stmt in stmts:
            if self._check_stmt(stmt, current):
                return True
        return False

    def _check_stmt(self, stmt: dict, scope: _Scope) -> bool:
        """Return True if control cannot reach the statement after this one."""
        kind = stmt["kind"]
        if kind == "let":
            typ = stmt["type"]
            # The initializer is checked *before* the name is bound: a let
            # is visible starting with the following statement.
            init_type = self._type_of(stmt["init"], scope)
            if init_type != typ:
                raise TypeCheckError(
                    f"let {stmt['name']!r}: declared {typ}, initializer is {init_type}"
                )
            scope.declare(stmt["name"], typ)
            return False
        if kind == "assign":
            name = stmt["target"]
            got = self._type_of(stmt["value"], scope)
            want = scope.lookup(name)
            if want is None:
                raise UndefinedSymbolError(
                    f"assignment to undeclared variable {name!r}"
                )
            if got != want:
                raise TypeCheckError(
                    f"assign to {name!r}: declared {want}, value is {got}"
                )
            return False
        if kind == "if":
            cond_t = self._type_of(stmt["cond"], scope)
            if cond_t != BOOL:
                raise TypeCheckError(f"if condition must be bool, got {cond_t}")
            then_done = self._check_stmts(stmt["then"], _Scope(scope))
            else_done = self._check_stmts(stmt["else"], _Scope(scope))
            return then_done and else_done
        if kind == "while":
            cond_t = self._type_of(stmt["cond"], scope)
            if cond_t != BOOL:
                raise TypeCheckError(f"while condition must be bool, got {cond_t}")
            # The loop may run zero times, so fallthrough always survives;
            # a return inside the body does not make the loop completing.
            self._check_stmts(stmt["body"], _Scope(scope))
            return False
        if kind == "return":
            value = stmt["value"]
            if self._ret_type == VOID:
                if value is not None:
                    got = self._type_of(value, scope)
                    raise TypeCheckError(
                        f"void function {self._func_name!r} returned {got}"
                    )
            else:
                if value is None:
                    raise TypeCheckError(
                        f"function {self._func_name!r} must return {self._ret_type}"
                    )
                got = self._type_of(value, scope)
                if got != self._ret_type:
                    raise TypeCheckError(
                        f"return in {self._func_name!r}: declared {self._ret_type}, got {got}"
                    )
            return True
        # block
        return self._check_stmts(stmt["body"], _Scope(scope))

    # -- expressions -------------------------------------------------------

    def _type_of(self, expr: dict, scope: _Scope) -> str:
        kind = expr["kind"]
        if kind == "int":
            return INT
        if kind == "bool":
            return BOOL
        if kind == "var":
            name = expr["name"]
            typ = scope.lookup(name)
            if typ is None:
                raise UndefinedSymbolError(f"reference to undeclared variable {name!r}")
            return typ
        if kind == "arith":
            lt = self._type_of(expr["left"], scope)
            rt = self._type_of(expr["right"], scope)
            if lt != INT or rt != INT:
                raise TypeCheckError(
                    f"arithmetic '{expr['op']}' requires int operands, got {lt} and {rt}"
                )
            return INT
        if kind == "compare":
            lt = self._type_of(expr["left"], scope)
            rt = self._type_of(expr["right"], scope)
            if lt != rt:
                raise TypeCheckError(
                    f"comparison '{expr['op']}' requires matching types, got {lt} and {rt}"
            )
            return BOOL
        if kind == "logical":
            lt = self._type_of(expr["left"], scope)
            rt = self._type_of(expr["right"], scope)
            if lt != BOOL or rt != BOOL:
                raise TypeCheckError(
                    f"boolean '{expr['op']}' requires bool operands, got {lt} and {rt}"
                )
            return BOOL
        # call
        name = expr["name"]
        func = self.functions.get(name)
        if func is None:
            raise UndefinedSymbolError(f"call to undeclared function {name!r}")
        args = expr["args"]
        if len(args) != len(func.param_types):
            raise TypeCheckError(
                f"call to {name!r}: expected {len(func.param_types)} "
                f"argument(s), got {len(args)}"
            )
        for i, (arg, want) in enumerate(zip(args, func.param_types)):
            got = self._type_of(arg, scope)
            if got != want:
                raise TypeCheckError(
                    f"call to {name!r} argument {i + 1}: expected {want}, got {got}"
                )
        if func.ret_type == VOID:
            raise TypeCheckError(
                f"call to void function {name!r} cannot be used as a value"
            )
        return func.ret_type


def analyze(ast: dict) -> None:
    """Run name resolution and type checking over a validated AST."""
    SemanticAnalyzer(ast).check()
