"""Value representation for Static Single Assignment (SSA) IR."""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tileforge.ir.types import Type
    from tileforge.ir.operation import Operation


class Value:
    """Represents an SSA value in TileForge IR."""
    def __init__(self, name: str, type_: Type, defining_op: Optional[Operation] = None):
        self.name: str = name
        self.type: Type = type_
        self.defining_op: Optional[Operation] = defining_op
        self.uses: List[Use] = []

    def add_use(self, op: Operation, operand_index: int) -> Use:
        use = Use(user=op, operand_index=operand_index, value=self)
        self.uses.append(use)
        return use

    def remove_use(self, op: Operation, operand_index: int) -> None:
        self.uses = [u for u in self.uses if not (u.user == op and u.operand_index == operand_index)]

    def replace_all_uses_with(self, new_value: Value) -> None:
        """Replace all uses of this SSA value with new_value (for optimizations)."""
        uses_to_replace = list(self.uses)
        for use in uses_to_replace:
            use.user.operands[use.operand_index] = new_value
            new_value.uses.append(Use(user=use.user, operand_index=use.operand_index, value=new_value))
        self.uses.clear()

    def is_used(self) -> bool:
        return len(self.uses) > 0

    def __repr__(self) -> str:
        return f"Value({self.name}: {self.type})"


class Use:
    """Represents a usage of an SSA value by an operation."""
    def __init__(self, user: Operation, operand_index: int, value: Value):
        self.user: Operation = user
        self.operand_index: int = operand_index
        self.value: Value = value

    def __repr__(self) -> str:
        return f"Use(user={self.user.op_type}, index={self.operand_index})"
