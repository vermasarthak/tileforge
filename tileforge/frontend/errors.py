"""Structured compiler errors for TileForge."""

class TileForgeError(Exception):
    """Base exception for all TileForge compiler errors."""
    def __init__(self, message: str, filename: str | None = None, line: int | None = None, column: int | None = None):
        self.message = message
        self.filename = filename
        self.line = line
        self.column = column
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        location_parts = []
        if self.filename:
            location_parts.append(self.filename)
        if self.line is not None:
            location_parts.append(str(self.line))
            if self.column is not None:
                location_parts.append(str(self.column))

        prefix = ":".join(location_parts)
        if prefix:
            return f"{prefix}: {self.message}"
        return self.message


class TileForgeSyntaxError(TileForgeError):
    """Raised when Python source cannot be parsed into valid TileForge AST."""
    pass


class UnsupportedSyntaxError(TileForgeError):
    """Raised when unsupported Python language features are encountered."""
    pass


class TypeCheckError(TileForgeError):
    """Raised when type checking or shape validation fails."""
    pass


class UndefinedSymbolError(TileForgeError):
    """Raised when an undefined variable or builtin symbol is referenced."""
    pass


class IRVerificationError(TileForgeError):
    """Raised when IR structure or SSA invariants fail verification."""
    pass


class LoweringError(TileForgeError):
    """Raised when lowering AST to IR fails."""
    pass


class InterpreterError(TileForgeError):
    """Raised when runtime IR interpretation fails."""
    pass
