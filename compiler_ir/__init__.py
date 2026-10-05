"""compiler-ir: AST -> non-SSA control-flow IR lowering pipeline.

Public entry points
-------------------
* :func:`lower_ir` / :func:`lower_module` -- accept a program AST made of
  plain dicts and lists, run name resolution and type checking, and return
  a traversable :class:`Module`.
* :func:`render_module` -- render a :class:`Module` to deterministic text.
* :func:`emit_ir` -- convenience wrapper combining the two.
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
    Return,
    Slot,
    Temp,
)
from .lowerer import lower_ir, lower_module
from .printer import render_module


def emit_ir(ast: dict) -> str:
    """Validate, analyze and lower ``ast`` and return its text IR."""
    return render_module(lower_module(ast))


__all__ = [
    "__version__",
    # entry points
    "lower_ir",
    "lower_module",
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
