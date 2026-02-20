"""Compiler-recognized builtins for TileForge kernel programming language."""

from typing import Any


def program_id(axis: int) -> Any:
    raise RuntimeError("tf.program_id is a TileForge compiler builtin and cannot be executed directly in Python.")


def arange(start: int, end: int) -> Any:
    raise RuntimeError("tf.arange is a TileForge compiler builtin and cannot be executed directly in Python.")


def load(pointer: Any, offsets: Any, mask: Any = None) -> Any:
    raise RuntimeError("tf.load is a TileForge compiler builtin and cannot be executed directly in Python.")


def store(pointer: Any, offsets: Any, value: Any, mask: Any = None) -> None:
    raise RuntimeError("tf.store is a TileForge compiler builtin and cannot be executed directly in Python.")


def zeros(shape: tuple, dtype: str) -> Any:
    raise RuntimeError("tf.zeros is a TileForge compiler builtin and cannot be executed directly in Python.")


def where(condition: Any, true_val: Any, false_val: Any) -> Any:
    raise RuntimeError("tf.where is a TileForge compiler builtin and cannot be executed directly in Python.")
