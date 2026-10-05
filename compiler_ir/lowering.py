"""Lowering of the checked program AST into non-SSA control-flow IR.

Block / temporary / slot numbers are allocated in one deterministic pass:
functions in AST order, then statements and expressions in depth-first
order. Nothing depends on dict key order, object addresses or set
iteration. Short-circuit ``and`` / ``or`` are expressed purely with
branches; the right operand is only evaluated on the path that needs it.
"""

from .ast_schema import ARITHMETIC_OPS, COMPARISON_OPS
from .ir import (
    BasicBlock,
    Function,
    Instruction,
    Module,
    Parameter,
    Slot,
    Temp,
    Terminator,
)

_BINOP_NAMES = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}


class _LocalScope:
    def __init__(self, parent=None):
        self.parent = parent
        self.names = {}

    def put(self, name, slot):
        self.names[name] = slot

    def get(self, name):
        scope = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None


class _FunctionLowerer:
    def __init__(self, fn_ast, return_types):
        self.ast = fn_ast
        self.return_types = return_types
        self.fn = Function(
            name=fn_ast["name"],
            params=[Parameter(p["name"], p["type"]) for p in fn_ast["params"]],
            return_type=fn_ast["return_type"],
        )
        self.current = None
        self._next_slot = 0
        self._next_temp = 0
        self._next_block = 0

    # -- identifier allocation -----------------------------------------

    def _new_slot(self, name, type_name):
        slot = Slot(self._next_slot, name, type_name)
        self._next_slot += 1
        self.fn.slots.append(slot)
        return slot

    def _new_temp(self, type_name):
        temp = Temp(self._next_temp, type_name)
        self._next_temp += 1
        self.fn.temps.append(temp)
        return temp

    def _new_block(self):
        block = BasicBlock(self._next_block)
        self._next_block += 1
        self.fn.blocks.append(block)
        return block

    # -- instruction emission (no-ops on a dead path) ------------------

    def _emit(self, instruction):
        if self.current is not None:
            self.current.instructions.append(instruction)

    def _const(self, value, type_name):
        if self.current is None:
            return None
        temp = self._new_temp(type_name)
        self.current.instructions.append(
            Instruction("const", temp, value=value))
        return temp

    def _load(self, slot):
        if self.current is None:
            return None
        temp = self._new_temp(slot.type)
        self.current.instructions.append(
            Instruction("load", temp, slot=slot))
        return temp

    def _store(self, slot, value):
        self._emit(Instruction("store", None, operands=[value], slot=slot))

    def _binop(self, op_name, left, right, type_name):
        if self.current is None:
            return None
        temp = self._new_temp(type_name)
        self.current.instructions.append(
            Instruction(op_name, temp, operands=[left, right]))
        return temp

    def _call(self, callee, args, type_name):
        if self.current is None:
            return None
        temp = self._new_temp(type_name)
        self.current.instructions.append(
            Instruction("call", temp, operands=list(args), callee=callee))
        return temp

    def _terminate(self, terminator):
        if self.current is not None and self.current.terminator is None:
            self.current.terminator = terminator
            self.current = None

    def _br(self, target):
        self._terminate(Terminator("br", targets=[target]))

    def _cbr(self, condition, if_true, if_false):
        self._terminate(Terminator(
            "cbr", condition=condition, targets=[if_true, if_false]))

    # -- entry point ----------------------------------------------------

    def lower(self):
        entry = self._new_block()
        self.current = entry

        scope = _LocalScope()
        for param in self.ast["params"]:
            # Parameters live in slots from the entry block on; no store
            # is needed because the calling convention fills them.
            scope.put(param["name"],
                      self._new_slot(param["name"], param["type"]))

        self._compile_stmts(self.ast["body"], scope)

        if self.current is not None:
            # Implicit return at the end of a void function. A non-void
            # function always has a terminating path by semantic analysis,
            # so this is only reached for void functions.
            self.current.terminator = Terminator("return")
            self.current = None

        self._prune_unreachable()
        return self.fn

    def _prune_unreachable(self):
        """Drop unreachable blocks (e.g. empty joins after two returning
        branches) and renumber the survivors in DFS order.

        Successors are pushed reversed so the true / fall-through edge is
        visited first, keeping the numbering independent of object
        addresses.
        """
        reachable = []
        seen = set()
        worklist = [self.fn.entry]
        while worklist:
            block = worklist.pop()
            if id(block) in seen:
                continue
            seen.add(id(block))
            reachable.append(block)
            if block.terminator is not None:
                worklist.extend(reversed(block.terminator.targets))

        self.fn.blocks = reachable
        for i, block in enumerate(self.fn.blocks):
            block.id = i

    # -- statements -----------------------------------------------------

    def _compile_stmts(self, stmts, scope):
        for stmt in stmts:
            if self.current is None:
                # A statement list can never become reachable again on its
                # own; enclosing control structures switch blocks itself.
                break
            self._compile_stmt(stmt, scope)

    def _compile_block(self, body, parent_scope):
        inner = _LocalScope(parent_scope)
        self._compile_stmts(body, inner)

    def _compile_stmt(self, stmt, scope):
        kind = stmt["kind"]

        if kind == "var_decl":
            slot = self._new_slot(stmt["name"], stmt["type"])
            if "init" in stmt:
                value = self._compile_expr(stmt["init"], scope)
                self._store(slot, value)
            scope.put(stmt["name"], slot)
            return

        if kind == "assign":
            value = self._compile_expr(stmt["value"], scope)
            self._store(scope.get(stmt["target"]), value)
            return

        if kind == "if":
            self._compile_if(stmt, scope)
            return

        if kind == "while":
            self._compile_while(stmt, scope)
            return

        if kind == "return":
            value = None
            if "value" in stmt and stmt["value"] is not None:
                value = self._compile_expr(stmt["value"], scope)
            self._terminate(Terminator("return", value=value))
            return

        if kind == "block":
            self._compile_block(stmt["body"], scope)
            return

    def _compile_if(self, stmt, scope):
        condition = self._compile_expr(stmt["cond"], scope)
        has_else = "else" in stmt

        then_block = self._new_block()
        if has_else:
            else_block = self._new_block()
        join_block = self._new_block()
        false_target = else_block if has_else else join_block
        self._cbr(condition, then_block, false_target)

        # Without an else clause the condition-false edge reaches the
        # join block directly.
        incoming = not has_else

        self.current = then_block
        self._compile_block(stmt["then"], scope)
        if self.current is not None:
            self._br(join_block)
            incoming = True

        if has_else:
            self.current = else_block
            self._compile_block(stmt["else"], scope)
            if self.current is not None:
                self._br(join_block)
                incoming = True

        if incoming:
            self.current = join_block
        # else: join is unreachable and empty; pruned after lowering.

    def _compile_while(self, stmt, scope):
        # _compile_stmts only dispatches statements on a live path.
        # Reuse the current block as the condition block when it is still
        # empty, otherwise jump to a fresh condition block first.
        if not self.current.instructions and self.current.terminator is None:
            cond_block = self.current
        else:
            cond_block = self._new_block()
            self._br(cond_block)
            self.current = cond_block

        body_block = self._new_block()
        exit_block = self._new_block()

        condition = self._compile_expr(stmt["cond"], scope)
        self._cbr(condition, body_block, exit_block)

        self.current = body_block
        self._compile_block(stmt["body"], scope)
        self._br(cond_block)  # back-edge (skipped if the body returned)

        self.current = exit_block

    # -- expressions ----------------------------------------------------

    def _compile_expr(self, expr, scope):
        if self.current is None:
            return None

        kind = expr["kind"]
        if kind == "int":
            return self._const(expr["value"], "int")
        if kind == "bool":
            return self._const(expr["value"], "bool")
        if kind == "var":
            return self._load(scope.get(expr["name"]))
        if kind == "call":
            args = [self._compile_expr(arg, scope) for arg in expr["args"]]
            return self._call(expr["name"], args,
                              self.return_types[expr["name"]])
        if kind == "binary":
            return self._compile_binary(expr, scope)
        # Structural validation already ruled every other kind out.
        raise AssertionError(f"unexpected expression kind {kind!r}")

    def _compile_binary(self, expr, scope):
        op = expr["op"]
        if op in ARITHMETIC_OPS:
            left = self._compile_expr(expr["left"], scope)
            right = self._compile_expr(expr["right"], scope)
            return self._binop(_BINOP_NAMES[op], left, right, "int")
        if op in COMPARISON_OPS:
            left = self._compile_expr(expr["left"], scope)
            right = self._compile_expr(expr["right"], scope)
            return self._binop(_BINOP_NAMES[op], left, right, "bool")

        # Short-circuit boolean operators: the result is stored through a
        # synthetic slot and merged back in a join block.
        is_and = op == "and"
        result_slot = self._new_slot(f".{op}{self._next_slot}", "bool")

        left = self._compile_expr(expr["left"], scope)

        rhs_block = self._new_block()
        short_block = self._new_block()
        join_block = self._new_block()
        if is_and:
            # true  -> evaluate the right operand
            # false -> short-circuit to false
            self._cbr(left, rhs_block, short_block)
            short_value = False
        else:
            # true  -> short-circuit to true
            # false -> evaluate the right operand
            self._cbr(left, short_block, rhs_block)
            short_value = True

        self.current = rhs_block
        right = self._compile_expr(expr["right"], scope)
        self._store(result_slot, right)
        self._br(join_block)

        self.current = short_block
        shortcut = self._const(short_value, "bool")
        self._store(result_slot, shortcut)
        self._br(join_block)

        self.current = join_block
        return self._load(result_slot)


def lower_program(root):
    """Lower a validated and semantically checked program AST."""
    return_types = {fn["name"]: fn["return_type"] for fn in root["functions"]}

    module = Module()
    for fn_ast in root["functions"]:
        module.functions.append(
            _FunctionLowerer(fn_ast, return_types).lower())
    return module
