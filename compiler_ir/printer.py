"""Deterministic plain-text rendering of the IR.

The output is a direct, fixed-order walk of the IR tree: functions in
module order, blocks in id order, phi nodes (SSA only) then instructions in
emission order.  Nothing is sorted by name or derived from object identity
-- phi entries are ordered by predecessor label, which is itself fixed -- so
a given module renders byte-identically across processes and repeated
conversions.

Non-SSA modules render exactly as before (``%tN`` temporaries, ``%vN``
slots).  SSA modules use a distinct representation: every value is a unique
``%N`` definition, parameters are rendered with their SSA value, no locals
section is emitted, and phi lines carry the result type plus one
``[block, value]`` pair per reachable predecessor.
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
    Phi,
    Return,
)


def _render_instruction(ins, value) -> str:
    if isinstance(ins, Const):
        literal = "true" if ins.value is True else "false" if ins.value is False else str(ins.value)
        return f"    {value(ins.dest)}: {ins.dest.type} = const {literal}"
    if isinstance(ins, Copy):
        return f"    {value(ins.dest)}: {ins.dest.type} = copy {value(ins.src)}"
    if isinstance(ins, BinOp):
        return (
            f"    {value(ins.dest)}: {ins.type} = {ins.kind} {ins.operator} "
            f"{value(ins.left)} {value(ins.right)}"
        )
    if isinstance(ins, Call):
        args = ", ".join(value(a) for a in ins.args)
        return f"    {value(ins.dest)}: {ins.type} = call {ins.name}({args})"
    raise AssertionError(f"unknown instruction: {ins!r}")  # pragma: no cover


def _render_phi(phi: Phi, value) -> str:
    sources = ", ".join(
        f"[{block.label}, {value(val)}]"
        for block, val in sorted(phi.entries.items(), key=lambda item: item[0].id)
    )
    return f"    {value(phi.dest)}: {phi.type} = phi {sources}"


def _render_terminator(term, value) -> str:
    if isinstance(term, Return):
        if term.value is None:
            return "    return"
        return f"    return {value(term.value)}"
    if isinstance(term, Jump):
        return f"    jump {term.target.label}"
    if isinstance(term, Branch):
        return (
            f"    br {value(term.cond)}, {term.true_target.label}, "
            f"{term.false_target.label}"
        )
    raise AssertionError(f"unknown terminator: {term!r}")  # pragma: no cover


def _render_function(func: Function) -> str:
    ssa = getattr(func, "ssa", False)

    if ssa:
        def value(v) -> str:
            return f"%{v.id}"

        params = ", ".join(
            f"{p.name}: {p.temp.type} @ {value(p.temp)}" for p in func.params
        )
    else:
        def value(v) -> str:
            return str(v)

        params = ", ".join(f"{p.name}: {p.slot.type} @ {value(p.slot)}"
                           for p in func.params)

    lines = [f"function {func.name}({params}) -> {func.ret_type} {{"]
    if not ssa and func.locals:
        lines.append("  locals:")
        for slot in func.locals:
            lines.append(f"    {value(slot)}: {slot.type}")
    for block in func.blocks:
        lines.append(f"  {block.label}:")
        if ssa:
            for phi in block.phis:
                lines.append(_render_phi(phi, value))
        for ins in block.instructions:
            lines.append(_render_instruction(ins, value))
        lines.append(_render_terminator(block.terminator, value))
    lines.append("}")
    return "\n".join(lines)


def render_module(module: Module) -> str:
    """Render a :class:`Module` to deterministic plain text."""
    parts = ["module"]
    parts.extend(_render_function(func) for func in module.functions)
    return "\n".join(parts) + "\n"
