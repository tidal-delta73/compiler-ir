"""Lowering from the validated, semantically checked AST to non-SSA IR.

Numbering is fully deterministic: functions are processed in AST order and
each function numbers blocks and temporaries from zero, allocating them in
the depth-first order in which statements and expressions are visited.
No dictionary key order, object identity or set iteration influences the
result, so the same AST produces byte-identical IR in every process.

Control flow
------------
* Every function gets an entry block ``b0``; parameters occupy the first
  local slots.
* ``if`` emits a conditional branch to true/false blocks and creates a
  merge block only when at least one branch can actually reach it; an
  unreachable, empty merge is never left behind.
* ``while`` emits a condition block, body block and exit block.
* ``return`` terminates its path; statements following a terminator on the
  same path produce no IR.
* Short-circuit ``and``/``or`` lower to control flow; the right operand is
  only evaluated on the path that needs it.
"""
from typing import Optional

from .ast_validate import validate_program
from .ir_nodes import (
    BinOp,
    Block,
    Branch,
    Call,
    Const,
    Copy,
    Function,
    Jump,
    Module,
    Parameter,
    Return,
    Slot,
    Temp,
)
from .semantics import analyze

INT = "int"
BOOL = "bool"


class _Scope:
    __slots__ = ("parent", "names")

    def __init__(self, parent: Optional["_Scope"] = None):
        self.parent = parent
        self.names: dict[str, Slot] = {}

    def declare(self, name: str, slot: Slot) -> None:
        self.names[name] = slot

    def lookup(self, name: str) -> Optional[Slot]:
        scope: Optional[_Scope] = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None


class _Lowerer:
    def __init__(self, ast: dict, callee_types: dict[str, str]):
        self.ast = ast
        self.callee_types = callee_types
        self.functions: list[Function] = []
        # Per-function state, reset in _lower_function.
        self.blocks: list[Block] = []
        self.local_slots: list[Slot] = []
        self.current: Optional[Block] = None
        self._block_id = 0
        self._temp_id = 0
        self._slot_id = 0

    # -- generic construction helpers -------------------------------------

    def new_block(self) -> Block:
        block = Block(self._block_id)
        self._block_id += 1
        self.blocks.append(block)
        return block

    def new_temp(self, typ: str) -> Temp:
        temp = Temp(self._temp_id, typ)
        self._temp_id += 1
        return temp

    def new_slot(self, typ: str) -> Slot:
        slot = Slot(self._slot_id, typ)
        self._slot_id += 1
        return slot

    def emit(self, instruction) -> None:
        if self.current is not None and not self.current.is_terminated:
            self.current.instructions.append(instruction)

    def terminate(self, terminator) -> None:
        if self.current is not None:
            self.current.terminator = terminator
            self.current = None

    # -- module / functions ------------------------------------------------

    def lower(self) -> Module:
        for func in self.ast["functions"]:
            self.functions.append(self._lower_function(func))
        return Module(self.functions)

    def _lower_function(self, func: dict) -> Function:
        self.blocks = []
        self.local_slots = []
        self.current = None
        self._block_id = 0
        self._temp_id = 0
        self._slot_id = 0

        scope = _Scope(None)
        params: list[Parameter] = []
        for param in func["params"]:
            slot = self.new_slot(param["type"])
            scope.declare(param["name"], slot)
            params.append(Parameter(param["name"], slot))

        entry = self.new_block()
        self.current = entry

        self._ret_type = func["ret_type"]
        self.lower_stmts(func["body"], scope)

        if self.current is not None:
            # Reachable fall-through is only possible for a void function;
            # the semantic pass guarantees non-void functions return on
            # every reachable path.
            self.current.terminator = Return(None)
            self.current = None

        return Function(
            func["name"],
            params,
            func["ret_type"],
            list(self.local_slots),
            self.blocks,
            entry,
        )

    # -- statements --------------------------------------------------------

    def lower_stmts(self, stmts: list, scope: _Scope) -> None:
        for stmt in stmts:
            if self.current is None:
                # A terminator already ended this path; later statements on
                # the same path produce no IR.
                return
            self.lower_stmt(stmt, scope)

    def lower_stmt(self, stmt: dict, scope: _Scope) -> None:
        kind = stmt["kind"]
        if kind == "let":
            value = self.lower_expr(stmt["init"], scope)
            slot = self.new_slot(stmt["type"])
            self.local_slots.append(slot)
            # The name is bound only after the initializer has been lowered.
            scope.declare(stmt["name"], slot)
            self.emit(Copy(slot, value))
            return
        if kind == "assign":
            value = self.lower_expr(stmt["value"], scope)
            slot = scope.lookup(stmt["target"])
            # Semantic analysis already proved this is bound.
            self.emit(Copy(slot, value))
            return
        if kind == "if":
            self._lower_if(stmt, scope)
            return
        if kind == "while":
            self._lower_while(stmt, scope)
            return
        if kind == "return":
            value = None
            if stmt["value"] is not None:
                value = self.lower_expr(stmt["value"], scope)
            self.terminate(Return(value))
            return
        # block
        self.lower_stmts(stmt["body"], _Scope(scope))

    def _lower_if(self, stmt: dict, scope: _Scope) -> None:
        cond = self.lower_expr(stmt["cond"], scope)
        # Blocks are created in DFS visitation order: true, false, merge.
        true_block = self.new_block()
        false_block = self.new_block()
        self.terminate(Branch(cond, true_block, false_block))

        self.current = true_block
        self.lower_stmts(stmt["then"], _Scope(scope))
        true_end = self.current

        self.current = false_block
        self.lower_stmts(stmt["else"], _Scope(scope))
        false_end = self.current

        if true_end is None and false_end is None:
            # Both branches return: there is no continuation, so no merge
            # block is created at all.
            return

        merge = self.new_block()
        if true_end is not None:
            true_end.terminator = Jump(merge)
        if false_end is not None:
            false_end.terminator = Jump(merge)
        self.current = merge

    def _lower_while(self, stmt: dict, scope: _Scope) -> None:
        cond_block = self.new_block()
        self.terminate(Jump(cond_block))

        self.current = cond_block
        cond = self.lower_expr(stmt["cond"], scope)
        body_block = self.new_block()
        exit_block = self.new_block()
        # A short-circuit condition terminates ``cond_block`` itself and
        # delivers its value in a later merge block; attach the loop branch
        # wherever evaluation actually ends, not blindly to cond_block.
        self.terminate(Branch(cond, body_block, exit_block))

        self.current = body_block
        self.lower_stmts(stmt["body"], _Scope(scope))
        body_end = self.current
        if body_end is not None:
            body_end.terminator = Jump(cond_block)
        # If the body always returns, the back edge simply does not exist;
        # the exit block remains reachable from the condition check.

        self.current = exit_block

    # -- expressions -------------------------------------------------------

    def lower_expr(self, expr: dict, scope: _Scope):
        """Return the value reference (Temp or Slot) holding the result."""
        kind = expr["kind"]
        if kind == "int":
            temp = self.new_temp(INT)
            self.emit(Const(temp, int(expr["value"])))
            return temp
        if kind == "bool":
            temp = self.new_temp(BOOL)
            self.emit(Const(temp, bool(expr["value"])))
            return temp
        if kind == "var":
            return scope.lookup(expr["name"])
        if kind == "arith":
            left = self.lower_expr(expr["left"], scope)
            right = self.lower_expr(expr["right"], scope)
            temp = self.new_temp(INT)
            self.emit(BinOp(temp, expr["op"], left, right, "arith", INT))
            return temp
        if kind == "compare":
            left = self.lower_expr(expr["left"], scope)
            right = self.lower_expr(expr["right"], scope)
            temp = self.new_temp(BOOL)
            self.emit(BinOp(temp, expr["op"], left, right, "compare", BOOL))
            return temp
        if kind == "logical":
            return self._lower_logical(expr, scope)
        # call
        args = [self.lower_expr(arg, scope) for arg in expr["args"]]
        ret_type = self.callee_types[expr["name"]]
        temp = self.new_temp(ret_type)
        self.emit(Call(temp, expr["name"], args, ret_type))
        return temp

    def _lower_logical(self, expr: dict, scope: _Scope) -> Temp:
        op = expr["op"]
        left = self.lower_expr(expr["left"], scope)

        # Blocks are created in DFS visitation order: true, false, merge.
        true_block = self.new_block()
        false_block = self.new_block()
        merge_block = self.new_block()
        self.terminate(Branch(left, true_block, false_block))

        # ``and`` evaluates the right operand on the true path and folds to
        # false on the false path; ``or`` mirrors that.
        rhs_on_true_path = op == "and"
        short_value = not rhs_on_true_path  # False for and, True for or

        result: Optional[Temp] = None

        def visit(block: Block, evaluates_rhs: bool) -> None:
            nonlocal result
            self.current = block
            if evaluates_rhs:
                right = self.lower_expr(expr["right"], scope)
                # The right operand may itself branch (nested short
                # circuit); the copy belongs to whatever block it ends in,
                # not necessarily to ``block``.
                end = self.current
                if result is None:
                    result = self.new_temp(BOOL)
                end.instructions.append(Copy(result, right))
                end.terminator = Jump(merge_block)
                self.current = None
            else:
                if result is None:
                    result = self.new_temp(BOOL)
                self.emit(Const(result, short_value))
                block.terminator = Jump(merge_block)
                self.current = None

        visit(true_block, rhs_on_true_path)
        visit(false_block, not rhs_on_true_path)

        self.current = merge_block
        assert result is not None
        return result


def lower_module(ast: dict) -> Module:
    """Validate, analyze and lower a plain-data program AST to a Module."""
    validate_program(ast)
    analyze(ast)
    callee_types = {func["name"]: func["ret_type"] for func in ast["functions"]}
    return _Lowerer(ast, callee_types).lower()


# Public alias.
lower_ir = lower_module
