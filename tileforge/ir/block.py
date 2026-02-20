"""Basic block representation for TileForge SSA IR."""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tileforge.ir.operation import Operation
    from tileforge.ir.function import Function


class Block:
    """Represents a basic block of sequential IR operations."""
    def __init__(self, name: str = "entry", parent_function: Optional[Function] = None):
        self.name: str = name
        self.operations: List[Operation] = []
        self.parent_function: Optional[Function] = parent_function

    def append_operation(self, op: Operation) -> None:
        op.parent_block = self
        self.operations.append(op)

    def insert_operation_before(self, target_op: Operation, new_op: Operation) -> None:
        idx = self.operations.index(target_op)
        new_op.parent_block = self
        self.operations.insert(idx, new_op)

    def __repr__(self) -> str:
        return f"Block({self.name}, ops={len(self.operations)})"
