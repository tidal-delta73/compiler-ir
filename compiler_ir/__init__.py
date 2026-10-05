"""compiler_ir: AST -> non-SSA control-flow IR lowering.

Public entry point: :func:`lower_module` accepts a program AST made of
plain dicts and lists and returns a traversable
:class:`~compiler_ir.ir.Module`; :func:`render_module` renders it as
deterministic plain text.
"""

from .ast_schema import validate_program
from .errors import (
    CompilerIRError,
    DuplicateSymbolError,
    InvalidAstError,
    MissingReturnError,
    TypeCheckError,
    UndefinedSymbolError,
)
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
from .lowering import lower_program
from .sema import check_program
from .text import render_function, render_module

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "lower_module",
    "render_module",
    "render_function",
    "validate_program",
    "check_program",
    "Module",
    "Function",
    "BasicBlock",
    "Instruction",
    "Terminator",
    "Temp",
    "Slot",
    "Parameter",
    "CompilerIRError",
    "InvalidAstError",
    "DuplicateSymbolError",
    "UndefinedSymbolError",
    "TypeCheckError",
    "MissingReturnError",
]


def lower_module(ast):
    """Validate, semantically check and lower a program AST.

    The AST is a plain data structure built from dicts and lists; see
    :mod:`compiler_ir.ast_schema` for the grammar. Returns a
    :class:`~compiler_ir.ir.Module`. Raises a subclass of
    :class:`InvalidAstError` (or :class:`MissingReturnError`) when the
    program is invalid.
    """
    validate_program(ast)
    check_program(ast)
    return lower_program(ast)
