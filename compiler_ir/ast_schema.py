"""Structural validation for the dict/list program AST.

The AST is a plain data structure:

Module   = {"functions": [Function, ...]}
Function = {"kind": "function", "name": str,
            "params": [{"name": str, "type": "int"|"bool"}, ...],
            "return_type": "int"|"bool"|"void",
            "body": [Statement, ...]}

Statements:
  var_decl {"kind": "var_decl", "name": str, "type": "int"|"bool",
            "init": Expression}            # "init" may be omitted
  assign   {"kind": "assign", "target": str, "value": Expression}
  if       {"kind": "if", "cond": Expression, "then": [Statement, ...],
            "else": [Statement, ...]}       # "else" may be omitted
  while    {"kind": "while", "cond": Expression, "body": [Statement, ...]}
  return   {"kind": "return", "value": Expression}  # "value" may be omitted
  block    {"kind": "block", "body": [Statement, ...]}

Expressions:
  int      {"kind": "int", "value": int}
  bool     {"kind": "bool", "value": bool}
  var      {"kind": "var", "name": str}
  binary   {"kind": "binary", "op": OP, "left": Expression, "right": Expression}
  call     {"kind": "call", "name": str, "args": [Expression, ...]}

Only structure is checked here; name resolution, type checking and return
path analysis live in :mod:`compiler_ir.sema`.
"""

from .errors import InvalidAstError

SCALAR_TYPES = ("int", "bool")
ALL_TYPES = SCALAR_TYPES + ("void",)

ARITHMETIC_OPS = ("+", "-", "*", "/")
COMPARISON_OPS = ("==", "!=", "<", "<=", ">", ">=")
BOOLEAN_OPS = ("and", "or")
BINARY_OPS = ARITHMETIC_OPS + COMPARISON_OPS + BOOLEAN_OPS


class _Path:
    """A locatable AST path; ``$`` denotes the module root."""

    __slots__ = ("text",)

    def __init__(self, text="$"):
        self.text = text

    def attr(self, name):
        return _Path(f"{self.text}.{name}")

    def index(self, i):
        return _Path(f"{self.text}[{i}]")

    def __str__(self):
        return self.text


def _fail(path, message):
    raise InvalidAstError(f"{path}: {message}")


def _is_int(value):
    # bool is a subclass of int; an integer literal must not carry a bool.
    return isinstance(value, int) and not isinstance(value, bool)


def _check_keys(node, allowed, path):
    extra = sorted(set(node) - set(allowed))
    if extra:
        _fail(path, f"unexpected field {extra[0]!r}")


def _require_dict(value, path, what):
    if not isinstance(value, dict):
        _fail(path, f"expected {what} object, got {type(value).__name__}")


def _require_field(node, field, path):
    if field not in node:
        _fail(path.attr(field), f"missing required field {field!r}")
    return node[field]


def _check_type_name(value, path, *, allow_void):
    if not isinstance(value, str):
        _fail(path, f"type name must be a string, got {type(value).__name__}")
    valid = ALL_TYPES if allow_void else SCALAR_TYPES
    if value not in valid:
        allowed = "|".join(valid)
        _fail(path, f"illegal type name {value!r} (expected {allowed})")


def validate_program(root):
    """Validate a whole program AST. Raises :class:`InvalidAstError`."""
    path = _Path()
    _require_dict(root, path, "program")
    allowed = {"kind", "functions"}
    _check_keys(root, allowed, path)
    if "kind" in root and root["kind"] != "module":
        _fail(path.attr("kind"), f"unknown module kind {root['kind']!r}")
    functions = _require_field(root, "functions", path)
    if not isinstance(functions, list):
        _fail(path.attr("functions"),
              f"expected a list of functions, got {type(functions).__name__}")
    for i, function in enumerate(functions):
        _validate_function(function, path.attr("functions").index(i))


def _validate_function(node, path):
    _require_dict(node, path, "function")
    _check_keys(node, {"kind", "name", "params", "return_type", "body"}, path)
    kind = _require_field(node, "kind", path)
    if kind != "function":
        _fail(path.attr("kind"), f"unknown node kind {kind!r}")
    name = _require_field(node, "name", path)
    if not isinstance(name, str) or not name:
        _fail(path.attr("name"), "function name must be a non-empty string")
    params = _require_field(node, "params", path)
    if not isinstance(params, list):
        _fail(path.attr("params"),
              f"expected a parameter list, got {type(params).__name__}")
    for i, param in enumerate(params):
        _validate_param(param, path.attr("params").index(i))
    ret = _require_field(node, "return_type", path)
    _check_type_name(ret, path.attr("return_type"), allow_void=True)
    body = _require_field(node, "body", path)
    _validate_statement_list(body, path.attr("body"))


def _validate_param(node, path):
    _require_dict(node, path, "parameter")
    _check_keys(node, {"name", "type"}, path)
    name = _require_field(node, "name", path)
    if not isinstance(name, str) or not name:
        _fail(path.attr("name"), "parameter name must be a non-empty string")
    type_name = _require_field(node, "type", path)
    _check_type_name(type_name, path.attr("type"), allow_void=False)


def _validate_statement_list(stmts, path):
    if not isinstance(stmts, list):
        _fail(path, f"expected a statement list, got {type(stmts).__name__}")
    for i, stmt in enumerate(stmts):
        _validate_statement(stmt, path.index(i))


_STATEMENT_KEYS = {
    "var_decl": {"kind", "name", "type", "init"},
    "assign": {"kind", "target", "value"},
    "if": {"kind", "cond", "then", "else"},
    "while": {"kind", "cond", "body"},
    "return": {"kind", "value"},
    "block": {"kind", "body"},
}


def _validate_statement(node, path):
    _require_dict(node, path, "statement")
    kind = node.get("kind")
    if not isinstance(kind, str):
        _fail(path, "statement node missing a string 'kind' field")
    allowed = _STATEMENT_KEYS.get(kind)
    if allowed is None:
        _fail(path.attr("kind"), f"unknown statement kind {kind!r}")
    _check_keys(node, allowed, path)

    if kind == "var_decl":
        name = _require_field(node, "name", path)
        if not isinstance(name, str) or not name:
            _fail(path.attr("name"),
                  "variable name must be a non-empty string")
        _check_type_name(_require_field(node, "type", path),
                         path.attr("type"), allow_void=False)
        if "init" in node:
            _validate_expr(node["init"], path.attr("init"))
    elif kind == "assign":
        target = _require_field(node, "target", path)
        if not isinstance(target, str) or not target:
            _fail(path.attr("target"),
                  "assignment target must be a non-empty name")
        _validate_expr(_require_field(node, "value", path),
                       path.attr("value"))
    elif kind == "if":
        _validate_expr(_require_field(node, "cond", path), path.attr("cond"))
        _validate_statement_list(_require_field(node, "then", path),
                                 path.attr("then"))
        if "else" in node:
            _validate_statement_list(node["else"], path.attr("else"))
    elif kind == "while":
        _validate_expr(_require_field(node, "cond", path), path.attr("cond"))
        _validate_statement_list(_require_field(node, "body", path),
                                 path.attr("body"))
    elif kind == "return":
        if "value" in node and node["value"] is not None:
            _validate_expr(node["value"], path.attr("value"))
    elif kind == "block":
        _validate_statement_list(_require_field(node, "body", path),
                                 path.attr("body"))


def _validate_expr(node, path):
    _require_dict(node, path, "expression")
    kind = node.get("kind")
    if not isinstance(kind, str):
        _fail(path, "expression node missing a string 'kind' field")

    if kind == "int":
        _check_keys(node, {"kind", "value"}, path)
        value = _require_field(node, "value", path)
        if not _is_int(value):
            _fail(path.attr("value"),
                  f"integer literal must be an int, got {type(value).__name__}")
    elif kind == "bool":
        _check_keys(node, {"kind", "value"}, path)
        value = _require_field(node, "value", path)
        if not isinstance(value, bool):
            _fail(path.attr("value"),
                  f"boolean literal must be a bool, got {type(value).__name__}")
    elif kind == "var":
        _check_keys(node, {"kind", "name"}, path)
        name = _require_field(node, "name", path)
        if not isinstance(name, str) or not name:
            _fail(path.attr("name"), "variable name must be a non-empty string")
    elif kind == "binary":
        _check_keys(node, {"kind", "op", "left", "right"}, path)
        op = _require_field(node, "op", path)
        if not isinstance(op, str) or op not in BINARY_OPS:
            _fail(path.attr("op"), f"unknown binary operator {op!r}")
        _validate_expr(_require_field(node, "left", path), path.attr("left"))
        _validate_expr(_require_field(node, "right", path), path.attr("right"))
    elif kind == "call":
        _check_keys(node, {"kind", "name", "args"}, path)
        name = _require_field(node, "name", path)
        if not isinstance(name, str) or not name:
            _fail(path.attr("name"), "callee name must be a non-empty string")
        args = _require_field(node, "args", path)
        if not isinstance(args, list):
            _fail(path.attr("args"),
                  f"expected an argument list, got {type(args).__name__}")
        for i, arg in enumerate(args):
            _validate_expr(arg, path.attr("args").index(i))
    else:
        _fail(path.attr("kind"), f"unknown expression kind {kind!r}")
