# compiler-ir

Compiler IR and optimisation pipeline.

Pure-Python (3.10+), no runtime dependencies.

## Usage

```bash
python3 -m compiler_ir version
python3 -m compiler_ir help
```

The command line still exposes only `version` and `help`; unknown commands
exit with code `2`. The compiler pipeline is used as a library.

## AST → non-SSA control-flow IR

`lower_module` (alias `lower_ir`) accepts a program AST built from plain
dicts and lists, performs name resolution and type checking, and returns a
traversable `Module`. `render_module` (or the `emit_ir` convenience
wrapper) turns it into deterministic plain text.

```python
from compiler_ir import lower_module, render_module, emit_ir

ast = {
    "functions": [
        {
            "name": "main",
            "params": [{"name": "n", "type": "int"}],
            "ret_type": "int",
            "body": [
                {"kind": "let", "name": "acc", "type": "int",
                 "init": {"kind": "int", "value": 0}},
                {"kind": "while",
                 "cond": {"kind": "compare", "op": "gt",
                          "left": {"kind": "var", "name": "n"},
                          "right": {"kind": "int", "value": 0}},
                 "body": [
                     {"kind": "assign", "target": "acc",
                      "value": {"kind": "arith", "op": "add",
                                "left": {"kind": "var", "name": "acc"},
                                "right": {"kind": "var", "name": "n"}}},
                     {"kind": "assign", "target": "n",
                      "value": {"kind": "arith", "op": "sub",
                                "left": {"kind": "var", "name": "n"},
                                "right": {"kind": "int", "value": 1}}},
                 ]},
                {"kind": "return", "value": {"kind": "var", "name": "acc"}},
            ],
        }
    ],
}

module = lower_module(ast)   # traversable Module
text = render_module(module)  # == emit_ir(ast)
```

### AST schema

* Program: `{"functions": [function, ...]}`
* Function: `{"name", "params": [{"name", "type"}], "ret_type", "body": [stmt]}`
  where `type`/`ret_type` are `int`, `bool`, or (`ret_type` only) `void`.
* Statements:
  * `{"kind": "let", "name", "type", "init": expr}`
  * `{"kind": "assign", "target", "value": expr}`
  * `{"kind": "if", "cond": expr, "then": [stmt], "else": [stmt]}`
  * `{"kind": "while", "cond": expr, "body": [stmt]}`
  * `{"kind": "return", "value": expr | null}`
  * `{"kind": "block", "body": [stmt]}`
* Expressions:
  * `{"kind": "int", "value": <int>}`, `{"kind": "bool", "value": <bool>}`
  * `{"kind": "var", "name"}`
  * `{"kind": "arith", "op": add|sub|mul|div|mod, "left", "right"}`
  * `{"kind": "compare", "op": eq|ne|lt|le|gt|ge, "left", "right"}`
  * `{"kind": "logical", "op": and|or, "left", "right"}` (short-circuit)
  * `{"kind": "call", "name", "args": [expr]}`

### Semantic rules

* A `let` is visible from the statement after its declaration; the
  initializer cannot refer to the variable itself.
* An inner same-name `let` shadows an outer one. A duplicate name in the
  same scope raises `DuplicateSymbolError`; referencing an undeclared name
  raises `UndefinedSymbolError`.
* Arithmetic accepts only `int`; `and`/`or` only `bool`; comparisons take
  matching operands and yield `bool`; assignments, returns and call
  arguments must match declared types. Violations raise `TypeCheckError`.
* A reachable path of a non-`void` function that ends without `return`
  raises `MissingReturnError`.
* Malformed AST (missing fields, unknown kinds/operators/types, wrong
  shape or literal Python type) raises `InvalidAstError` carrying a
  locatable path such as `module.functions[0].body[2].cond`.

### IR text

The text contains each function signature, its local slots, and stably
numbered basic blocks (`b0`, `b1`, …) of ordinary instructions followed by
one explicit terminator (`return`, `jump`, `br`). Temporaries (`%tN`) and
slots (`%vN`) carry their types. `if` lowers to true/false blocks plus a
merge block that is only created when reachable; `while` lowers to
condition/body/exit blocks; short-circuit `and`/`or` branch so the right
operand is evaluated only when needed.

Numbers are assigned in function order and statement/expression
depth-first order — never from dict key order, object identity or set
iteration — so a given AST produces byte-identical text across processes.

## Non-SSA IR → SSA IR

`to_ssa` takes the `Module` returned by `lower_module` and returns a *new*
`Module` in static single-assignment form; the input module is left
unchanged. Anything that is not a `Module` raises `TypeError`.

```python
from compiler_ir import lower_module, render_module, to_ssa

module = lower_module(ast)      # non-SSA Module
ssa = to_ssa(module)            # new SSA Module; module is untouched
print(render_module(ssa))       # renders phi nodes ahead of instructions
```

In the SSA module:

* Every parameter and instruction result is a unique SSA definition; all
  slot (`%vN`) reads and writes are gone — a slot write becomes pure
  renaming, a slot read becomes the value currently reaching it. The
  multi-path writes produced by short-circuit `and`/`or` are split into
  one definition per path plus a phi at the merge.
* A `Phi` appears at a control-flow join only when a value with several
  reachable definitions is actually observed afterwards. Phis sit ahead of
  the block's ordinary instructions and record one `(predecessor, value)`
  pair per reachable in-edge, ordered by predecessor block label:
  `%t4: int = phi [b1: %t2, b2: %t3]`. Unreachable predecessors never
  contribute an edge, and phis whose in-edges all carry the same value —
  or whose result is never used — are removed.
* Loop-carried variables get a phi at the loop header that takes the
  entering value from before the loop and the updated value from the
  back edge.
* Function order, block labels, block order and terminators are preserved;
  calls, arithmetic, comparisons, branch targets, returns and side-effect
  order are equivalent to the input module.

SSA value numbers are assigned per function from zero and are a pure
function of parameter order, block order and intra-block phi/instruction
order, so converting and rendering the same input twice is byte-identical,
and applying `to_ssa` to an already-SSA module is a fixed point (no new
phis, no renumbering).

## Tests

```bash
python3 -m unittest discover -s tests -v
```
