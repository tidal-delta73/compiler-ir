"""Standalone executor for instruction-selected SSA target text.

This is *test support* for the execution-level pass-order regression suite
in ``test_pass_order_execution.py``.  It is **not** part of the compiler
package and adds no production dependency: it is plain standard-library
Python, launched by the suite as a real operating-system process.

Why this exists
---------------
The language has no native code emitter; instruction selection is the
deterministic :func:`compiler_ir.render_module` plain-text emission, the
compiler's final stage.  The execution suite therefore:

1. compiles one legal source AST through the existing public entry points
   (``lower_module`` -> a legal permutation of ``to_ssa`` /
   ``fold_constants`` / ``eliminate_dead_code`` -> ``render_module``);
2. writes the rendered text to its own temporary target file;
3. *executes* that artifact in an isolated OS process and captures the
   real process exit status, standard output and standard error.

This module parses **only** rendered SSA text (every legal schedule in the
execution suite starts with ``to_ssa`` and ends in rendering) and executes
it directly.  It never imports the compiler: the target artifact is
self-contained, so no compiler object (symbol tables, temporary numbers,
analysis caches or pass state) can ride along into the executed program.

The observed output channel
---------------------------
The language has no I/O statement.  The only side effect a program can
perform is a function call, so calls to the reserved ``emit`` function
``emit(int) -> int`` are the standard-output surrogate: each executed call
writes its integer argument on its own line to stdout.  ``emit`` is the
identity, so its result is usable in surrounding arithmetic.

Process contract
----------------
* ``main`` returning a non-zero int or ``true`` -> exit status 1;
  returning zero or ``false`` -> 0; ``void main`` falling through -> 0.
* A runtime division/modulo by zero writes one line to stderr and exits 3.
* Any malformed target line, unknown instruction/terminator/operator,
  missing or duplicate SSA definition, bad block edge, type mismatch,
  non-bool branch, arity mismatch, non-integer argument or exhausted
  step/depth budget writes a diagnostic to stderr and exits 2.

Every target the compiler is expected to produce is well-formed and
terminating, so an exit 2/3 in the execution suite marks a broken order.
The exit status only carries 0/1 for ``main``; arbitrary integer results
are exchanged between functions inside the process (a concrete-value
recursive evaluator), while ``emit`` output still crosses the real
stdout boundary in execution order.
"""
from __future__ import annotations

import re
import sys


_STEP_LIMIT = 1_000_000
_DEPTH_LIMIT = 1_000


def _trunc_div(a, b):
    if b == 0:
        raise ZeroDivisionError
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def _trunc_mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a - _trunc_div(a, b) * b


_ARITH = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _trunc_div,
    "mod": _trunc_mod,
}

_COMPARE = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def _fail(message):
    """Abort with status 2, distinct from every normal result."""
    sys.stderr.write("target-runner: malformed target: " + str(message)
                     + "\n")
    sys.stderr.flush()
    sys.exit(2)


def _trap(function_name, operator):
    sys.stderr.write(
        f"target-runner: runtime fault: {function_name}: {operator} by "
        "zero\n")
    sys.stderr.flush()
    sys.exit(3)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r"^function (\w+)\((.*)\) -> (int|bool|void) \{$")
_BLOCK_RE = re.compile(r"^  (b\d+):$")
_PARAM_RE = re.compile(r"^(\w+): (int|bool) @ %(\d+)$")
_CONST_RE = re.compile(r"^    %(\d+): (int|bool) = const (.+)$")
_BINOP_RE = re.compile(
    r"^    %(\d+): (int|bool) = (arith|compare) (\w+) %(\d+) %(\d+)$")
_CALL_RE = re.compile(
    r"^    %(\d+): (int|bool) = call (\w+)\((.*)\)$")
_PHI_RE = re.compile(
    r"^    (%\d+): (?:int|bool) = phi ((?:\[b\d+, %\d+\](?:, )?)+)$")
_PHI_ENTRY_RE = re.compile(r"\[(b\d+), (%\d+)\]")
_RETURN_RE = re.compile(r"^    return(?: (%\d+))?$")
_JUMP_RE = re.compile(r"^    jump (b\d+)$")
_BRANCH_RE = re.compile(r"^    br (%\d+), (b\d+), (b\d+)$")
_VALUE_RE = re.compile(r"^%(\d+)$")
_INT_RE = re.compile(r"^-?\d+$")


def _vid(token):
    m = _VALUE_RE.match(token)
    if not m:
        _fail(f"bad value reference {token!r}")
    return int(m.group(1))


def _literal(token):
    if token == "true":
        return True
    if token == "false":
        return False
    if _INT_RE.match(token):
        return int(token)
    _fail(f"bad literal {token!r}")


def _parse_params(text, lineno):
    if text == "":
        return []
    params = []
    for piece in text.split(", "):
        m = _PARAM_RE.match(piece)
        if not m:
            _fail(f"line {lineno}: bad parameter spec {piece!r}")
        name, typ, vid = m.groups()
        params.append((name, typ, int(vid)))
    return params


def _parse(text):
    """Parse rendered SSA text into {name: function dict}."""
    lines = text.splitlines()
    if not lines or lines[0].rstrip("\n") != "module":
        _fail("target text must start with 'module'")

    programs = {}
    fn = None
    block = None
    phi_zone = False

    for lineno, raw in enumerate(lines[1:], start=2):
        line = raw

        m = _FUNC_RE.match(line)
        if m:
            if fn is not None:
                _fail(f"line {lineno}: nested function")
            name, param_text, ret_type = m.groups()
            if name in programs:
                _fail(f"line {lineno}: duplicate function {name!r}")
            fn = {
                "name": name,
                "ret_type": ret_type,
                "params": _parse_params(param_text, lineno),
                "blocks": {},
                "order": [],
            }
            programs[name] = fn
            block = None
            continue

        if line == "}":
            if fn is None:
                _fail(f"line {lineno}: stray '}}'")
            if block is not None and block["terminator"] is None:
                _fail(f"line {lineno}: block {block['label']} has no "
                      "terminator")
            fn = None
            block = None
            continue

        if fn is None:
            if line.strip() == "":
                continue
            _fail(f"line {lineno}: unexpected line {line!r}")

        m = _BLOCK_RE.match(line)
        if m:
            label = m.group(1)
            if label in fn["blocks"]:
                _fail(f"line {lineno}: duplicate block {label}")
            if block is not None and block["terminator"] is None:
                _fail(f"line {lineno}: previous block has no terminator")
            block = {"label": label, "phis": [], "instructions": [],
                     "terminator": None}
            fn["blocks"][label] = block
            fn["order"].append(label)
            phi_zone = True
            continue

        if block is None:
            _fail(f"line {lineno}: instruction outside a block: {line!r}")

        m = _PHI_RE.match(line)
        if m:
            if not phi_zone:
                _fail(f"line {lineno}: phi after ordinary instructions")
            dest = _vid(m.group(1))
            entries = tuple(
                (em.group(1), _vid(em.group(2)))
                for em in _PHI_ENTRY_RE.finditer(m.group(2))
            )
            if not entries:
                _fail(f"line {lineno}: empty phi")
            preds = [e[0] for e in entries]
            if len(preds) != len(set(preds)):
                _fail(f"line {lineno}: phi lists a predecessor twice")
            block["phis"].append((dest, entries))
            continue
        phi_zone = False

        m = _CONST_RE.match(line)
        if m:
            dest, typ, token = m.groups()
            value = _literal(token)
            if typ == "int" and (not isinstance(value, int)
                                 or isinstance(value, bool)):
                _fail(f"line {lineno}: int const holds non-int literal")
            if typ == "bool" and not isinstance(value, bool):
                _fail(f"line {lineno}: bool const holds non-bool literal")
            block["instructions"].append(("const", int(dest), value))
            continue

        m = _BINOP_RE.match(line)
        if m:
            dest, _typ, kind, op, left, right = m.groups()
            table = _ARITH if kind == "arith" else _COMPARE
            if op not in table:
                _fail(f"line {lineno}: unknown {kind} operator {op!r}")
            block["instructions"].append(
                (kind, int(dest), op, int(left), int(right)))
            continue

        m = _CALL_RE.match(line)
        if m:
            dest, typ, callee, args_text = m.groups()
            arg_ids = (() if args_text == "" else
                       tuple(_vid(x.strip())
                             for x in args_text.split(",")))
            block["instructions"].append(
                ("call", int(dest), typ, callee, arg_ids))
            continue

        m = _RETURN_RE.match(line)
        if m:
            token = m.group(1)
            value_id = None if token is None else _vid(token)
            if (fn["ret_type"] == "void") != (value_id is None):
                _fail(f"line {lineno}: return shape disagrees with "
                      f"declared {fn['ret_type']} return")
            block["terminator"] = ("return", value_id)
            continue

        m = _JUMP_RE.match(line)
        if m:
            block["terminator"] = ("jump", m.group(1))
            continue

        m = _BRANCH_RE.match(line)
        if m:
            cond, t, f = m.groups()
            block["terminator"] = ("branch", _vid(cond), t, f)
            continue

        _fail(f"line {lineno}: unrecognized line {line!r}")

    if fn is not None:
        _fail("target text ends inside a function")
    if not programs:
        _fail("target text has no functions")
    if "main" not in programs:
        _fail("target text has no main function")

    for program in programs.values():
        _validate(program, programs)
    return programs


def _validate(fn, programs):
    """Cross-reference checks that line-local parsing cannot perform."""
    defined = {p[2] for p in fn["params"]}
    if len(defined) != len(fn["params"]):
        _fail(f"{fn['name']}: duplicate parameter SSA id")

    for label in fn["order"]:
        block = fn["blocks"][label]
        if block["terminator"] is None:
            _fail(f"{fn['name']}:{label}: missing terminator")
        for dest, entries in block["phis"]:
            if dest in defined:
                _fail(f"{fn['name']}:{label}: duplicate definition %{dest}")
            defined.add(dest)
            for pred, val_id in entries:
                if pred not in fn["blocks"]:
                    _fail(f"{fn['name']}:{label}: phi from unknown "
                          f"predecessor {pred}")
        for ins in block["instructions"]:
            dest = ins[1]
            if dest in defined:
                _fail(f"{fn['name']}:{label}: duplicate definition %{dest}")
            defined.add(dest)

    def require(value_id, where):
        if value_id not in defined:
            _fail(f"{fn['name']}:{where}: reference to undefined "
                  f"%{value_id}")

    for label in fn["order"]:
        block = fn["blocks"][label]
        for dest, entries in block["phis"]:
            for pred, val_id in entries:
                require(val_id, label)
        for ins in block["instructions"]:
            if ins[0] == "const":
                pass
            elif ins[0] in ("arith", "compare"):
                require(ins[3], label)
                require(ins[4], label)
            elif ins[0] == "call":
                _typ, callee, arg_ids = ins[2], ins[3], ins[4]
                for arg in arg_ids:
                    require(arg, label)
                if callee == "emit":
                    if _typ != "int" or len(arg_ids) != 1:
                        _fail(f"{fn['name']}:{label}: emit must be "
                              "emit(int) -> int")
                elif callee not in programs:
                    _fail(f"{fn['name']}:{label}: call to unknown "
                          f"function {callee!r}")
                else:
                    callee_params = programs[callee]["params"]
                    if len(arg_ids) != len(callee_params):
                        _fail(f"{fn['name']}:{label}: arity mismatch in "
                              f"call to {callee!r}")
        term = block["terminator"]
        if term[0] == "return":
            if term[1] is not None:
                require(term[1], label)
        elif term[0] == "jump":
            if term[1] not in fn["blocks"]:
                _fail(f"{fn['name']}:{label}: jump to unknown "
                      f"{term[1]}")
        elif term[0] == "branch":
            require(term[1], label)
            for target in (term[2], term[3]):
                if target not in fn["blocks"]:
                    _fail(f"{fn['name']}:{label}: branch to unknown "
                          f"{target}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self, programs):
        self.programs = programs

    def invoke_main(self, arguments):
        """Run main with concrete argument values; return its exit status."""
        value = self._invoke("main", arguments, depth=0, steps=[_STEP_LIMIT])
        main_type = self.programs["main"]["ret_type"]
        if main_type == "void":
            return 0
        if main_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                _fail("main: int function produced non-int")
            return 1 if value != 0 else 0
        if isinstance(value, bool):
            return 1 if value else 0
        _fail("main: bool function produced non-bool")

    def _invoke(self, name, arguments, depth, steps):
        if depth > _DEPTH_LIMIT:
            _fail(f"{name}: call depth limit exceeded")
        fn = self.programs[name]
        if len(arguments) != len(fn["params"]):
            _fail(f"{name}: arity mismatch ({len(arguments)} for "
                  f"{len(fn['params'])})")

        values = {
            param[2]: arg for param, arg in zip(fn["params"], arguments)
        }
        label = fn["order"][0]
        previous = None

        while True:
            steps[0] -= 1
            if steps[0] <= 0:
                _fail(f"{name}: step budget exhausted (non-terminating?)")

            block = fn["blocks"][label]

            # All incoming values are read off the actual predecessor edge
            # before any phi result is bound.
            pending = []
            for dest, entries in block["phis"]:
                if previous is None:
                    _fail(f"{name}:{label}: phi reached on entry")
                incoming = dict(entries)
                if previous not in incoming:
                    _fail(f"{name}:{label}: phi has no incoming value "
                          f"from {previous}")
                pending.append((dest, values[incoming[previous]]))
            values.update(pending)

            for ins in block["instructions"]:
                self._execute(fn, values, ins, depth, steps)

            term = block["terminator"]
            if term[0] == "return":
                if term[1] is None:
                    return None
                return values[term[1]]

            if term[0] == "jump":
                previous, label = label, term[1]
                continue

            cond = values[term[1]]
            if not isinstance(cond, bool):
                _fail(f"{name}:{label}: branch on non-bool {cond!r}")
            previous, label = label, (term[2] if cond else term[3])

    def _execute(self, fn, values, ins, depth, steps):
        tag = ins[0]
        if tag == "const":
            values[ins[1]] = ins[2]
            return

        if tag in ("arith", "compare"):
            _kind, dest, op, left_id, right_id = ins
            left = values[left_id]
            right = values[right_id]
            if tag == "arith":
                if not isinstance(left, int) or isinstance(left, bool) or \
                        not isinstance(right, int) or isinstance(right, bool):
                    _fail(f"{fn['name']}: arithmetic on non-int operands")
                try:
                    result = _ARITH[op](left, right)
                except ZeroDivisionError:
                    _trap(fn["name"], op)
            else:
                if type(left) is not type(right):
                    _fail(f"{fn['name']}: comparison between mismatched "
                          "types")
                result = _COMPARE[op](left, right)
            values[dest] = result
            return

        if tag == "call":
            _kind, dest, _typ, callee, arg_ids = ins
            args = [values[arg] for arg in arg_ids]
            if callee == "emit":
                arg = args[0]
                if not isinstance(arg, int) or isinstance(arg, bool):
                    _fail(f"{fn['name']}: emit of non-int value")
                sys.stdout.write(str(arg) + "\n")
                sys.stdout.flush()
                values[dest] = arg
                return
            callee_fn = self.programs[callee]
            for param, arg in zip(callee_fn["params"], args):
                ptype = param[1]
                if ptype == "int" and (not isinstance(arg, int)
                                       or isinstance(arg, bool)):
                    _fail(f"{fn['name']}->{callee}: int argument got "
                          f"{arg!r}")
                if ptype == "bool" and not isinstance(arg, bool):
                    _fail(f"{fn['name']}->{callee}: bool argument got "
                          f"{arg!r}")
            values[dest] = self._invoke(callee, args, depth + 1, steps)
            return

        _fail(f"{fn['name']}: unknown instruction {ins!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _read_arguments(path):
    with open(path, "r", encoding="utf-8") as fh:
        tokens = [line.rstrip("\n") for line in fh if line != "\n"]
    return [_literal(token) for token in tokens]


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(
            "usage: target_runner.py <target-file> <arguments-file>\n")
        return 2
    try:
        with open(argv[1], "r", encoding="utf-8") as fh:
            text = fh.read()
        arguments = _read_arguments(argv[2])
    except OSError as exc:
        _fail(f"cannot read inputs: {exc}")

    programs = _parse(text)
    main_fn = programs["main"]
    if len(arguments) != len(main_fn["params"]):
        _fail(f"main expects {len(main_fn['params'])} argument(s), got "
              f"{len(arguments)}")
    for param, arg in zip(main_fn["params"], arguments):
        pname, ptype, _vid_ = param
        if ptype == "int" and (not isinstance(arg, int)
                               or isinstance(arg, bool)):
            _fail(f"argument {pname!r}: expected int, got {arg!r}")
        if ptype == "bool" and not isinstance(arg, bool):
            _fail(f"argument {pname!r}: expected bool, got {arg!r}")

    return _Engine(programs).invoke_main(arguments)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
