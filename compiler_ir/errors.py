"""Exception hierarchy for the AST -> IR lowering pipeline."""


class CompilerIRError(Exception):
    """Base class for all compiler_ir errors."""


class InvalidAstError(CompilerIRError):
    """The input value is not a well formed program AST.

    The message carries a locatable AST path (``$`` is the module root).
    """


class DuplicateSymbolError(InvalidAstError):
    """A name is declared twice in the same scope."""


class UndefinedSymbolError(InvalidAstError):
    """A name is referenced without a visible declaration."""


class TypeCheckError(InvalidAstError):
    """A construct is well formed but violates the type rules."""


class MissingReturnError(InvalidAstError):
    """A non-void function has a reachable path that does not return."""
