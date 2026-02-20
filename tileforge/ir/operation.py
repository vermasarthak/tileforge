"""Operation representation for TileForge SSA IR."""

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tileforge.ir.value import Value
    from tileforge.ir.block import Block


class OpType:
    CONSTANT = "tf.constant"
    ADD = "tf.add"
    SUB = "tf.sub"
    MUL = "tf.mul"
    DIV = "tf.div"
    CMP = "tf.cmp"
    PROGRAM_ID = "tf.program_id"
    ARANGE = "tf.arange"
    LOAD = "tf.load"
    STORE = "tf.store"
    WHERE = "tf.where"
    RETURN = "tf.return"


# Operations without side-effects (candidates for CSE/DCE)
PURE_OPERATIONS = {
    OpType.CONSTANT,
    OpType.ADD,
    OpType.SUB,
    OpType.MUL,
    OpType.DIV,
    OpType.CMP,
    OpType.PROGRAM_ID,
    OpType.ARANGE,
    OpType.WHERE,
}


class Operation:
    """Represents a single IR instruction/operation."""
    def __init__(
        self,
        op_type: str,
        operands: Optional[List[Value]] = None,
        results: Optional[List[Value]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        parent_block: Optional[Block] = None,
    ):
        self.op_type: str = op_type
        self.operands: List[Value] = operands if operands is not None else []
        self.results: List[Value] = results if results is not None else []
        self.attributes: Dict[str, Any] = attributes if attributes is not None else {}
        self.parent_block: Optional[Block] = parent_block

        # Set defining_op on results
        for res in self.results:
            res.defining_op = self

        # Register uses on operands
        for idx, opnd in enumerate(self.operands):
            opnd.add_use(self, idx)

    def is_pure(self) -> bool:
        return self.op_type in PURE_OPERATIONS

    def is_terminator(self) -> bool:
        return self.op_type == OpType.RETURN

    def erase(self) -> None:
        """Remove this operation from its parent block and unregister uses."""
        for idx, opnd in enumerate(self.operands):
            opnd.remove_use(self, idx)
        if self.parent_block is not None:
            if self in self.parent_block.operations:
                self.parent_block.operations.remove(self)
            self.parent_block = None

    def __repr__(self) -> str:
        results_str = ", ".join(r.name for r in self.results)
        operands_str = ", ".join(o.name for o in self.operands)
        attrs_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
        
        res_part = f"{results_str} = " if results_str else ""
        attr_part = f" {{{attrs_str}}}" if attrs_str else ""
        return f"{res_part}{self.op_type}({operands_str}){attr_part}"
