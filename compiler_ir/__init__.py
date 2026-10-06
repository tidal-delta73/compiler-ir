"""compiler-ir: AST -> non-SSA control-flow IR lowering pipeline.

Public entry points
-------------------
* :func:`lower_ir` / :func:`lower_module` -- accept a program AST made of
  plain dicts and lists, run name resolution and type checking, and return
  a traversable :class:`Module`.
* :func:`to_ssa` -- convert a lowered ``Module`` to a new SSA ``Module``.
* :func:`render_module` -- render a :class:`Module` to deterministic text.
* :func:`emit_ir` -- convenience wrapper combining lowering and rendering.
"""
__version__ = "0.1.0"

from .errors import (
    CompilerError,
    DuplicateSymbolError,
    InvalidAstError,
    MissingReturnError,
    TypeCheckError,
    UndefinedSymbolError,
)
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
    Phi,
    Return,
    Slot,
    Temp,
)
from .lowerer import lower_ir, lower_module
from .printer import render_module
from .ssa import to_ssa


def emit_ir(ast: dict) -> str:
    """Validate, analyze and lower ``ast`` and return its text IR."""
    return render_module(lower_module(ast))


__all__ = [
    "__version__",
    # entry points
    "lower_ir",
    "lower_module",
    "to_ssa",
    "render_module",
    "emit_ir",
    # IR object model
    "Module",
    "Function",
    "Block",
    "Parameter",
    "Temp",
    "Slot",
    "Const",
    "Copy",
    "BinOp",
    "Call",
    "Phi",
    "Return",
    "Jump",
    "Branch",
    # errors
    "CompilerError",
    "InvalidAstError",
    "DuplicateSymbolError",
    "UndefinedSymbolError",
    "TypeCheckError",
    "MissingReturnError",
]
