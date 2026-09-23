"""Lexical symbol table and scope management for TileForge compiler."""

from __future__ import annotations

from typing import Dict, Optional

from tileforge.frontend.errors import TypeCheckError, UndefinedSymbolError
from tileforge.ir.types import Type


class Symbol:
    def __init__(self, name: str, type_: Type, is_arg: bool = False):
        self.name: str = name
        self.type: Type = type_
        self.is_arg: bool = is_arg

    def __repr__(self) -> str:
        return f"Symbol({self.name}: {self.type})"


class SymbolTable:
    """Scoped symbol table mapping variable names to Symbol objects."""
    def __init__(self, parent: Optional[SymbolTable] = None):
        self.parent: Optional[SymbolTable] = parent
        self.symbols: Dict[str, Symbol] = {}

    def declare(self, name: str, type_: Type, is_arg: bool = False, line: int = 0, column: int = 0) -> Symbol:
        if name in self.symbols:
            if is_arg:
                raise TypeCheckError(f"Duplicate argument name '{name}'", line=line, column=column)
        symbol = Symbol(name, type_, is_arg=is_arg)
        self.symbols[name] = symbol
        return symbol

    def lookup(self, name: str, line: int = 0, column: int = 0) -> Symbol:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent is not None:
            return self.parent.lookup(name, line, column)
        raise UndefinedSymbolError(f"Undefined symbol '{name}'", line=line, column=column)

    def contains(self, name: str) -> bool:
        if name in self.symbols:
            return True
        if self.parent is not None:
            return self.parent.contains(name)
        return False
