"""Deterministic plain-text rendering of the non-SSA IR.

The text of an identical AST is byte-for-byte stable across processes:
blocks, temps and slots are printed in their allocated numeric order and
no object addresses or set/dict iteration ever surfaces.
"""

from .ir import Module


def _literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_instruction(instr):
    if instr.op == "const":
        return f"{instr.dest.name}: {instr.dest.type} = const {_literal(instr.value)}"
    if instr.op == "load":
        return (f"{instr.dest.name}: {instr.dest.type} = load "
                f"{instr.slot.label}")
    if instr.op == "store":
        return f"store {instr.slot.label} {instr.operands[0].name}"
    if instr.op == "call":
        args = ", ".join(f"{op.name}: {op.type}" for op in instr.operands)
        return (f"{instr.dest.name}: {instr.dest.type} = call "
                f"@{instr.callee}({args})")
    # binary arithmetic / comparison ops
    left, right = instr.operands
    return (f"{instr.dest.name}: {instr.dest.type} = {instr.op} "
            f"{left.name} {right.name}")


def _format_terminator(term):
    if term.op == "return":
        if term.value is None:
            return "return"
        return f"return {term.value.name}: {term.value.type}"
    if term.op == "br":
        return f"br {term.targets[0].label}"
    # conditional branch: true target first, false target second
    return (f"cbr {term.condition.name} "
            f"{term.targets[0].label} {term.targets[1].label}")


def render_function(function):
    lines = []
    params = ", ".join(f"{p.name}: {p.type}" for p in function.params)
    lines.append(f"func @{function.name}({params}) -> {function.return_type} {{")

    for slot in function.slots:
        lines.append(f"  slot {slot.label}: {slot.type} ; {slot.name}")

    for block in function.blocks:
        lines.append(f"{block.label}:")
        for instr in block.instructions:
            lines.append(f"  {_format_instruction(instr)}")
        if block.terminator is not None:
            lines.append(f"  {_format_terminator(block.terminator)}")
    lines.append("}")
    return "\n".join(lines)


def render_module(module: Module) -> str:
    """Render a :class:`~compiler_ir.ir.Module` as deterministic text."""
    parts = [render_function(fn) for fn in module.functions]
    return ("\n\n".join(parts) + "\n") if parts else ""
