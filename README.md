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

## Non-SSA IR → pruned SSA

`to_ssa` takes the `Module` returned by `lower_module` and returns a brand
new, traversable and renderable SSA `Module`; the input is never mutated.
Hand-built modules are not a supported input; passing any non-`Module`
object raises `TypeError`. Feeding an SSA module back in copies it to an
equivalent SSA module with no new phi nodes and unchanged numbering.

```python
from compiler_ir import lower_module, to_ssa, render_module

module = lower_module(ast)
ssa = to_ssa(module)              # new Module; module is left untouched
print(render_module(ssa))
```

Properties of the result:

* Every parameter and every instruction result is a unique SSA value; all
  `Slot` reads/writes and the multi-path duplicate writes of short-circuit
  results are eliminated (an SSA `let`/assignment copy folds into a name,
  so no `copy` instruction remains).
* A `phi` is emitted at a control-flow join only when a value genuinely has
  more than one reachable incoming definition that is used afterwards.
  Phi nodes precede the block's ordinary instructions, and each phi lists
  one `[block, value]` pair per reachable predecessor in ascending
  predecessor-label order. Unreachable predecessors never appear; phis
  whose value is identical on every edge (or whose result is unused,
  including closed phi-only cycles) are pruned away.
* Both `if` branches, the missing `else`, nested blocks, loop back edges
  and short-circuit `and`/`or` are renamed correctly; a value carried
  across loop iterations gets the backedge value at the loop header and
  the entry value on the zero-iteration path.
* Function calls, arithmetic, comparisons, branch targets, return values
  and side-effect ordering are equivalent to the input. Function order,
  original basic-block labels, block order and terminators are preserved.
* SSA value numbers restart at zero per function and are fixed by
  parameter order, then block order, then (within a block) phi order
  followed by ordinary-instruction order — never by set/dict iteration or
  object identity, so converting and rendering the same module repeatedly
  is byte-identical.

The SSA text uses unique `%N` values (no `%tN`/`%vN`), renders parameters
with their SSA value, omits the `locals:` section, and shows typed phi
lines, e.g.:

```text
  b3:
    %7: int = phi [b1, %4], [b2, %6]
    return %7
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

## Tests

```bash
python3 -m unittest discover -s tests -v
```
