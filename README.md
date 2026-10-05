# compiler-ir

Compiler IR and optimisation pipeline.

Pure-Python (3.10+), no runtime dependencies.

## Usage

```bash
python3 -m compiler_ir version
python3 -m compiler_ir help
```

The CLI intentionally keeps the `version` and `help` commands only; the
compiler pipeline is a Python API.

## AST → non-SSA control-flow IR

`lower_module` accepts a program built from plain dicts and lists,
performs structural validation, name resolution, type checking and
return-path analysis, then returns a traversable `Module`:

```python
import compiler_ir as cir

ast = {
    "functions": [
        {
            "kind": "function",
            "name": "add",
            "params": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "body": [
                {"kind": "return",
                 "value": {"kind": "binary", "op": "+",
                           "left": {"kind": "var", "name": "x"},
                           "right": {"kind": "int", "value": 1}}},
            ],
        }
    ]
}

module = cir.lower_module(ast)        # compiler_ir.ir.Module
print(cir.render_module(module))
```

### AST grammar

Types are `"int"`, `"bool"` and, for function results, `"void"`.

```
Module   {"functions": [Function, ...]}
Function {"kind": "function", "name": str,
          "params": [{"name": str, "type": "int"|"bool"}, ...],
          "return_type": "int"|"bool"|"void",
          "body": [Statement, ...]}

var_decl {"kind": "var_decl", "name": str, "type": "int"|"bool",
          "init": Expression?}
assign   {"kind": "assign", "target": str, "value": Expression}
if       {"kind": "if", "cond": Expression,
          "then": [Statement, ...], "else"?: [Statement, ...]}
while    {"kind": "while", "cond": Expression, "body": [Statement, ...]}
return   {"kind": "return", "value"?: Expression}   # omit value in void
block    {"kind": "block", "body": [Statement, ...]}

int      {"kind": "int", "value": int}
bool     {"kind": "bool", "value": bool}
var      {"kind": "var", "name": str}
binary   {"kind": "binary", "op": OP,
          "left": Expression, "right": Expression}
call     {"kind": "call", "name": str, "args": [Expression, ...]}
```

Operators: arithmetic `+ - * /` (int → int), comparisons
`== != < <= > >=` (int operands → bool), short-circuit `and or`
(bool operands → bool). `and` / `or` lower to branches; the right
operand is never evaluated unconditionally.

### Rules

- A local declaration is visible from the statement after it; an inner
  declaration shadows the outer name.
- Duplicate declarations raise `DuplicateSymbolError`; references to
  undeclared names raise `UndefinedSymbolError`.
- Assignments, return values and call arguments must match declared
  types; arithmetic accepts only `int`, boolean operators only `bool`,
  and comparisons produce `bool`. Violations raise `TypeCheckError`.
- A non-void function whose control flow can reach the end without a
  return raises `MissingReturnError`.
- Malformed structure (missing fields, unknown kinds, illegal type
  names, wrong value shapes) raises `InvalidAstError` whose message
  carries a locatable path such as `$.functions[0].body[2].cond`.

### IR and text format

Each function owns an entry basic block, a slot per named local
(parameters included), and a temporary per expression result. `if`
lowers to true/false blocks plus a join when the flow continues;
`while` lowers to condition, body and exit blocks; `return` is an
explicit terminator and statements after it produce no IR. Empty,
unreachable joins are removed.

The text format lists the function signature, slots, and the numbered
blocks with their instructions and explicit terminators (`br`, `cbr`,
`return`). Numbering follows function order then depth-first statement
and expression order, so the same AST renders to byte-identical text in
every process regardless of hash seed or collection iteration order.
