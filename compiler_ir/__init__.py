"""compiler-ir: AST -> control-flow IR lowering pipeline.

Public entry points
-------------------
* :func:`lower_ir` / :func:`lower_module` -- accept a program AST made of
  plain dicts and lists, run name resolution and type checking, and return
  a traversable non-SSA :class:`Module`.
* :func:`to_ssa` -- convert a :class:`Module` into a brand new, traversable
  pruned-SSA :class:`Module` without mutating the input.
* :func:`eliminate_dead_code` -- remove definitions unreachable from the
  SSA roots (terminator operands and calls) into a brand new SSA
  :class:`Module`, without mutating the input.
* :func:`fold_constants` -- propagate known literals along SSA def-use
  chains and fold constant arithmetic/comparison BinOps and same-literal
  phis into a brand new SSA :class:`Module`, without mutating the input.
* :func:`render_module` -- render either flavor of :class:`Module` to
  deterministic text.
* :func:`optimize_module` -- run a whole optimization pass sequence
  (default: ``ssa, fold, dce, ssa``) in one call, returning a brand new
  :class:`Module`.
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
from .dce import eliminate_dead_code
from .folding import fold_constants
from .optimize import DEFAULT_PASSES, optimize_module


def emit_ir(ast: dict) -> str:
    """Validate, analyze and lower ``ast`` and return its text IR."""
    return render_module(lower_module(ast))


__all__ = [
    "__version__",
    # entry points
    "lower_ir",
    "lower_module",
    "to_ssa",
    "eliminate_dead_code",
    "fold_constants",
    "optimize_module",
    "DEFAULT_PASSES",
    "render_module",
    "emit_ir",
    # IR object model
    "Module",
    "Function",
    "Block",
    "Parameter",
    "Phi",
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
