"""Backend abstraction interface for TileForge compilers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from tileforge.ir.module import Module


class CompiledKernel(ABC):
    """Abstract base class for a compiled kernel ready for execution."""
    @abstractmethod
    def launch(self, grid: Tuple[int, ...], args: List[Any]) -> None:
        """Launches the compiled kernel with the given grid bounds and buffer arguments."""
        pass


class Backend(ABC):
    """Abstract base class for TileForge target backends (CPU, Metal, etc.)."""
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend identifier."""
        pass

    @abstractmethod
    def lower(self, module: Module) -> Any:
        """Lowers high-level TileForge SSA IR module into backend IR or representation."""
        pass

    @abstractmethod
    def compile(self, lowered_module: Any, func_name: str) -> CompiledKernel:
        """Compiles lowered module into executable compute pipeline."""
        pass
