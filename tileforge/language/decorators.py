"""Decorators for marking TileForge kernel functions."""

import inspect
from typing import Callable, Any


class KernelFunction:
    """Wrapper holding a compiled or source-captured TileForge kernel function."""
    def __init__(self, fn: Callable[..., Any]):
        self.fn: Callable[..., Any] = fn
        self.__name__: str = fn.__name__
        self.__doc__: str | None = fn.__doc__
        self._is_tileforge_kernel: bool = True
        try:
            self._source_code: str = inspect.getsource(fn)
        except Exception:
            self._source_code = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"Kernel function '{self.__name__}' cannot be called directly as standard Python. "
            "Use TileForge compiler/driver to compile and launch it on CPU reference runtime."
        )


def kernel(fn: Callable[..., Any]) -> KernelFunction:
    """Decorator marking a Python function as a TileForge kernel."""
    return KernelFunction(fn)
