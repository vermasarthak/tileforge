"""Compiler-recognized builtins for TileForge kernel programming language."""

from typing import Any


def program_id(axis: int) -> Any:
    raise RuntimeError("tf.program_id is a TileForge compiler builtin and cannot be executed directly in Python.")


def arange(start: int, end: int) -> Any:
    raise RuntimeError("tf.arange is a TileForge compiler builtin and cannot be executed directly in Python.")


def range(start: int, end: int) -> Any:
    raise RuntimeError("tf.range is a TileForge compiler builtin and cannot be executed directly in Python.")


def load(pointer: Any, offsets: Any, mask: Any = None) -> Any:
    raise RuntimeError("tf.load is a TileForge compiler builtin and cannot be executed directly in Python.")


def store(pointer: Any, offsets: Any, value: Any, mask: Any = None) -> None:
    raise RuntimeError("tf.store is a TileForge compiler builtin and cannot be executed directly in Python.")


def zeros(shape: Any, dtype: str = "f32") -> Any:
    raise RuntimeError("tf.zeros is a TileForge compiler builtin and cannot be executed directly in Python.")


def where(condition: Any, true_val: Any, false_val: Any) -> Any:
    raise RuntimeError("tf.where is a TileForge compiler builtin and cannot be executed directly in Python.")


def logical_and(a: Any, b: Any) -> Any:
    raise RuntimeError("tf.logical_and is a TileForge compiler builtin and cannot be executed directly in Python.")


def logical_or(a: Any, b: Any) -> Any:
    raise RuntimeError("tf.logical_or is a TileForge compiler builtin and cannot be executed directly in Python.")


def sum(x: Any, axis: Any = None) -> Any:
    raise RuntimeError("tf.sum is a TileForge compiler builtin and cannot be executed directly in Python.")


def max(x: Any, axis: Any = None) -> Any:
    raise RuntimeError("tf.max is a TileForge compiler builtin and cannot be executed directly in Python.")


def dot(a: Any, b: Any) -> Any:
    raise RuntimeError("tf.dot is a TileForge compiler builtin and cannot be executed directly in Python.")


def reshape(x: Any, shape: Any) -> Any:
    raise RuntimeError("tf.reshape is a TileForge compiler builtin and cannot be executed directly in Python.")


def expand_dims(x: Any, axis: int) -> Any:
    raise RuntimeError("tf.expand_dims is a TileForge compiler builtin and cannot be executed directly in Python.")


def broadcast_to(x: Any, shape: Any) -> Any:
    raise RuntimeError("tf.broadcast_to is a TileForge compiler builtin and cannot be executed directly in Python.")
