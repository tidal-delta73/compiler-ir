"""Structural validation of the dict/list program AST.

The AST is plain data produced by callers; this pass only checks shape
(required fields, node kinds, type names, list lengths, literal types) and
never performs name resolution or type checking.  Every failure raises
:class:`InvalidAstError` carrying a locatable path into the AST.
"""
from typing import Any

from .errors import InvalidAstError

INT = "int"
BOOL = "bool"
VOID = "void"
VALUE_TYPES = (INT, BOOL)
ALL_TYPES = (INT, BOOL, VOID)

ARITH_OPS = ("add", "sub", "mul", "div", "mod")
COMPARE_OPS = ("eq", "ne", "lt", "le", "gt", "ge")
LOGICAL_OPS = ("and", "or")

STATEMENT_KINDS = ("let", "assign", "if", "while", "return", "block")
EXPRESSION_KINDS = (
    "int", "bool", "var", "arith", "compare", "logical", "call",
)


def _fail(path: str, message: str) -> None:
    raise InvalidAstError(message, path)


def _is_ident(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) > 0
        and (value[0].isalpha() or value[0] == "_")
        and all(c.isalnum() or c == "_" for c in value)
    )


def _check_value_type(value: Any, path: str) -> None:
    if not isinstance(value, str) or value not in VALUE_TYPES:
        _fail(path, f"illegal type name {value!r}: expected 'int' or 'bool'")


def validate_program(ast: Any) -> dict:
    """Validate a whole program AST and return it unchanged."""
    path = "module"
    if not isinstance(ast, dict):
        _fail(path, f"program AST must be a dict, got {type(ast).__name__}")
    if "functions" not in ast:
        _fail(path, "missing required field 'functions'")
    funcs = ast["functions"]
    if not isinstance(funcs, list):
        _fail(f"{path}.functions", "must be a list")
    for i, func in enumerate(funcs):
        _validate_function(func, f"{path}.functions[{i}]")
    return ast


def _validate_function(func: Any, path: str) -> None:
    if not isinstance(func, dict):
        _fail(path, f"function must be a dict, got {type(func).__name__}")
    for field in ("name", "params", "ret_type", "body"):
        if field not in func:
            _fail(path, f"missing required field '{field}'")
    name = func["name"]
    if not _is_ident(name):
        _fail(f"{path}.name", f"invalid function name {name!r}")
    params = func["params"]
    if not isinstance(params, list):
        _fail(f"{path}.params", "must be a list")
    for i, param in enumerate(params):
        _validate_param(param, f"{path}.params[{i}]")
    ret = func["ret_type"]
    if not isinstance(ret, str) or ret not in ALL_TYPES:
        _fail(
            f"{path}.ret_type",
            f"illegal type name {ret!r}: expected 'int', 'bool' or 'void'",
        )
    _validate_block_body(func["body"], f"{path}.body")
    extra = set(func) - {"name", "params", "ret_type", "body"}
    if extra:
        _fail(path, f"unknown function field(s): {', '.join(sorted(extra))}")


def _validate_param(param: Any, path: str) -> None:
    if not isinstance(param, dict):
        _fail(path, f"parameter must be a dict, got {type(param).__name__}")
    for field in ("name", "type"):
        if field not in param:
            _fail(path, f"missing required field '{field}'")
    if not _is_ident(param["name"]):
        _fail(f"{path}.name", f"invalid parameter name {param['name']!r}")
    _check_value_type(param["type"], f"{path}.type")
    extra = set(param) - {"name", "type"}
    if extra:
        _fail(path, f"unknown parameter field(s): {', '.join(sorted(extra))}")


def _validate_block_body(body: Any, path: str) -> None:
    if not isinstance(body, list):
        _fail(path, f"statement list must be a list, got {type(body).__name__}")
    for i, stmt in enumerate(body):
        validate_statement(stmt, f"{path}[{i}]")


def validate_statement(stmt: Any, path: str) -> None:
    if not isinstance(stmt, dict):
        _fail(path, f"statement must be a dict, got {type(stmt).__name__}")
    if "kind" not in stmt:
        _fail(path, "missing required field 'kind'")
    kind = stmt["kind"]
    if kind not in STATEMENT_KINDS:
        _fail(path, f"unknown statement kind {kind!r}")
    if kind == "let":
        for field in ("name", "type", "init"):
            if field not in stmt:
                _fail(path, f"missing required field '{field}'")
        if not _is_ident(stmt["name"]):
            _fail(f"{path}.name", f"invalid variable name {stmt['name']!r}")
        _check_value_type(stmt["type"], f"{path}.type")
        _validate_expr(stmt["init"], f"{path}.init")
        extra = set(stmt) - {"kind", "name", "type", "init"}
    elif kind == "assign":
        for field in ("target", "value"):
            if field not in stmt:
                _fail(path, f"missing required field '{field}'")
        if not _is_ident(stmt["target"]):
            _fail(f"{path}.target", f"invalid variable name {stmt['target']!r}")
        _validate_expr(stmt["value"], f"{path}.value")
        extra = set(stmt) - {"kind", "target", "value"}
    elif kind == "if":
        for field in ("cond", "then", "else"):
            if field not in stmt:
                _fail(path, f"missing required field '{field}'")
        _validate_expr(stmt["cond"], f"{path}.cond")
        _validate_block_body(stmt["then"], f"{path}.then")
        _validate_block_body(stmt["else"], f"{path}.else")
        extra = set(stmt) - {"kind", "cond", "then", "else"}
    elif kind == "while":
        for field in ("cond", "body"):
            if field not in stmt:
                _fail(path, f"missing required field '{field}'")
        _validate_expr(stmt["cond"], f"{path}.cond")
        _validate_block_body(stmt["body"], f"{path}.body")
        extra = set(stmt) - {"kind", "cond", "body"}
    elif kind == "return":
        if "value" not in stmt:
            _fail(path, "missing required field 'value'")
        if stmt["value"] is not None:
            _validate_expr(stmt["value"], f"{path}.value")
        extra = set(stmt) - {"kind", "value"}
    else:  # block
        if "body" not in stmt:
            _fail(path, "missing required field 'body'")
        _validate_block_body(stmt["body"], f"{path}.body")
        extra = set(stmt) - {"kind", "body"}
    if extra:
        _fail(path, f"unknown {kind} statement field(s): {', '.join(sorted(extra))}")


def _validate_expr(expr: Any, path: str) -> None:
    if not isinstance(expr, dict):
        _fail(path, f"expression must be a dict, got {type(expr).__name__}")
    if "kind" not in expr:
        _fail(path, "missing required field 'kind'")
    kind = expr["kind"]
    if kind not in EXPRESSION_KINDS:
        _fail(path, f"unknown expression kind {kind!r}")
    if kind == "int":
        if "value" not in expr:
            _fail(path, "missing required field 'value'")
        # bool is a subclass of int: reject it explicitly.
        if not isinstance(expr["value"], int) or isinstance(expr["value"], bool):
            _fail(
                f"{path}.value",
                f"int literal must be an integer, got {type(expr['value']).__name__}",
            )
        extra = set(expr) - {"kind", "value"}
    elif kind == "bool":
        if "value" not in expr:
            _fail(path, "missing required field 'value'")
        if not isinstance(expr["value"], bool):
            _fail(
                f"{path}.value",
                f"bool literal must be true/false, got {type(expr['value']).__name__}",
            )
        extra = set(expr) - {"kind", "value"}
    elif kind == "var":
        if "name" not in expr:
            _fail(path, "missing required field 'name'")
        if not _is_ident(expr["name"]):
            _fail(f"{path}.name", f"invalid variable name {expr['name']!r}")
        extra = set(expr) - {"kind", "name"}
    elif kind in ("arith", "compare", "logical"):
        for field in ("op", "left", "right"):
            if field not in expr:
                _fail(path, f"missing required field '{field}'")
        op = expr["op"]
        if kind == "arith":
            allowed = ARITH_OPS
        elif kind == "compare":
            allowed = COMPARE_OPS
        else:
            allowed = LOGICAL_OPS
        if not isinstance(op, str) or op not in allowed:
            _fail(
                f"{path}.op",
                f"unknown {kind} operator {op!r}: expected one of {', '.join(allowed)}",
            )
        _validate_expr(expr["left"], f"{path}.left")
        _validate_expr(expr["right"], f"{path}.right")
        extra = set(expr) - {"kind", "op", "left", "right"}
    else:  # call
        for field in ("name", "args"):
            if field not in expr:
                _fail(path, f"missing required field '{field}'")
        if not _is_ident(expr["name"]):
            _fail(f"{path}.name", f"invalid callee name {expr['name']!r}")
        args = expr["args"]
        if not isinstance(args, list):
            _fail(f"{path}.args", "must be a list")
        for i, arg in enumerate(args):
            _validate_expr(arg, f"{path}.args[{i}]")
        extra = set(expr) - {"kind", "name", "args"}
    if extra:
        _fail(path, f"unknown {kind} expression field(s): {', '.join(sorted(extra))}")
