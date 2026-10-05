"""Exception hierarchy for the compiler-ir pipeline."""


class CompilerError(Exception):
    """Base class for all compiler errors."""


class InvalidAstError(CompilerError):
    """The program AST is structurally malformed.

    ``path`` is a human-readable, locatable path into the AST (for example
    ``module.functions[2].body[1].cond``); it is also appended to the
    exception message.
    """

    def __init__(self, message: str, path: str = "module"):
        self.path = path
        super().__init__(f"{message} (at {path})")


class DuplicateSymbolError(CompilerError):
    """A name is declared twice in the same scope."""


class UndefinedSymbolError(CompilerError):
    """A name is used before (or without) being declared."""


class TypeCheckError(CompilerError):
    """The program is well formed but not well typed."""


class MissingReturnError(CompilerError):
    """A non-``void`` function has a reachable path without a ``return``."""
