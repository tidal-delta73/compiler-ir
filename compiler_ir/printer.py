"""Deterministic plain-text rendering of the IR.

The output is a direct, fixed-order walk of the IR tree: functions in
module order, blocks in id order, instructions in emission order.  Nothing
is sorted by name or derived from object identity, so a given AST renders
byte-identically across processes.
"""
from .ir_nodes import (
    BinOp,
    Branch,
    Call,
    Const,
    Copy,
    Function,
    Jump,
    Module,
    Return,
)


def _value(value) -> str:
    return str(value)


def _render_instruction(ins) -> str:
    if isinstance(ins, Const):
        literal = "true" if ins.value is True else "false" if ins.value is False else str(ins.value)
        return f"    {ins.dest}: {ins.dest.type} = const {literal}"
    if isinstance(ins, Copy):
        return f"    {ins.dest}: {ins.dest.type} = copy {_value(ins.src)}"
    if isinstance(ins, BinOp):
        return (
            f"    {ins.dest}: {ins.type} = {ins.kind} {ins.operator} "
            f"{_value(ins.left)} {_value(ins.right)}"
        )
    if isinstance(ins, Call):
        args = ", ".join(_value(a) for a in ins.args)
        return f"    {ins.dest}: {ins.type} = call {ins.name}({args})"
    raise AssertionError(f"unknown instruction: {ins!r}")  # pragma: no cover


def _render_terminator(term) -> str:
    if isinstance(term, Return):
        if term.value is None:
            return "    return"
        return f"    return {_value(term.value)}"
    if isinstance(term, Jump):
        return f"    jump {term.target.label}"
    if isinstance(term, Branch):
        return (
            f"    br {_value(term.cond)}, {term.true_target.label}, "
            f"{term.false_target.label}"
        )
    raise AssertionError(f"unknown terminator: {term!r}")  # pragma: no cover


def _render_function(func: Function) -> str:
    params = ", ".join(f"{p.name}: {p.slot.type} @ {p.slot}" for p in func.params)
    lines = [f"function {func.name}({params}) -> {func.ret_type} {{"]
    if func.locals:
        lines.append("  locals:")
        for slot in func.locals:
            lines.append(f"    {slot}: {slot.type}")
    for block in func.blocks:
        lines.append(f"  {block.label}:")
        for ins in block.instructions:
            lines.append(_render_instruction(ins))
        lines.append(_render_terminator(block.terminator))
    lines.append("}")
    return "\n".join(lines)


def render_module(module: Module) -> str:
    """Render a :class:`Module` to deterministic plain text."""
    parts = ["module"]
    parts.extend(_render_function(func) for func in module.functions)
    return "\n\n".join(parts) + "\n"
