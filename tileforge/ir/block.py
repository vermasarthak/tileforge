"""Basic block representation for TileForge SSA IR with block arguments and CFG edges."""

from __future__ import annotations
from typing import List, Optional, Set, TYPE_CHECKING
from tileforge.ir.value import Value

if TYPE_CHECKING:
    from tileforge.ir.operation import Operation
    from tileforge.ir.function import Function


class Block:
    """Represents a basic block of sequential IR operations with block arguments."""
    def __init__(self, name: str = "entry", parent_function: Optional[Function] = None, args: Optional[List[Value]] = None):
        self.name: str = name
        self.operations: List[Operation] = []
        self.parent_function: Optional[Function] = parent_function
        self.args: List[Value] = args if args is not None else []

    def add_argument(self, arg: Value) -> None:
        self.args.append(arg)

    def append_operation(self, op: Operation) -> None:
        op.parent_block = self
        self.operations.append(op)

    def insert_operation_before(self, target_op: Operation, new_op: Operation) -> None:
        idx = self.operations.index(target_op)
        new_op.parent_block = self
        self.operations.insert(idx, new_op)

    @property
    def terminator(self) -> Optional[Operation]:
        if self.operations and self.operations[-1].is_terminator():
            return self.operations[-1]
        return None

    @property
    def successors(self) -> List[Block]:
        term = self.terminator
        if term is not None:
            return term.successors
        return []

    @property
    def predecessors(self) -> List[Block]:
        if self.parent_function is None:
            return []
        preds: List[Block] = []
        for b in self.parent_function.blocks:
            if self in b.successors:
                preds.append(b)
        return preds

    def __repr__(self) -> str:
        args_str = ", ".join(f"{a.name}: {a.type}" for a in self.args)
        arg_part = f"({args_str})" if args_str else ""
        return f"Block(^{self.name}{arg_part}, ops={len(self.operations)})"
