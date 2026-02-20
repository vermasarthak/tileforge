"""Module representation for TileForge SSA IR top-level container."""

from __future__ import annotations
from typing import List, Optional, Dict
from tileforge.ir.function import Function


class Module:
    """Top-level container holding compiled functions."""
    def __init__(self, name: str = "main_module"):
        self.name: str = name
        self.functions: List[Function] = []

    def add_function(self, func: Function) -> None:
        func.parent_module = self
        self.functions.append(func)

    def get_function(self, name: str) -> Optional[Function]:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None

    def __repr__(self) -> str:
        return f"Module({self.name}, functions={len(self.functions)})"
