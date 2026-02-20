"""IRBuilder for constructing TileForge SSA IR operations."""

from __future__ import annotations
from typing import List, Optional, Any, Tuple
from tileforge.ir.types import Type, PrimitiveType, TensorType, PointerType, I1, I32, promote_types, compare_types
from tileforge.ir.value import Value
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.block import Block


class IRBuilder:
    """Helper class to build operations inside basic blocks with SSA value allocation."""
    def __init__(self, block: Optional[Block] = None):
        self.block: Optional[Block] = block
        self._value_counter: int = 0

    def set_insertion_point(self, block: Block) -> None:
        self.block = block

    def new_value_name(self) -> str:
        name = f"%{self._value_counter}"
        self._value_counter += 1
        return name

    def create_constant(self, val: Any, type_: Type) -> Value:
        res = Value(self.new_value_name(), type_)
        op = Operation(OpType.CONSTANT, operands=[], results=[res], attributes={"value": val})
        if self.block:
            self.block.append_operation(op)
        return res

    def create_add(self, lhs: Value, rhs: Value) -> Value:
        res_type = promote_types(lhs.type, rhs.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.ADD, operands=[lhs, rhs], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_sub(self, lhs: Value, rhs: Value) -> Value:
        res_type = promote_types(lhs.type, rhs.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.SUB, operands=[lhs, rhs], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_mul(self, lhs: Value, rhs: Value) -> Value:
        res_type = promote_types(lhs.type, rhs.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.MUL, operands=[lhs, rhs], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_div(self, lhs: Value, rhs: Value) -> Value:
        res_type = promote_types(lhs.type, rhs.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.DIV, operands=[lhs, rhs], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_cmp(self, predicate: str, lhs: Value, rhs: Value) -> Value:
        res_type = compare_types(lhs.type, rhs.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.CMP, operands=[lhs, rhs], results=[res], attributes={"predicate": predicate})
        if self.block:
            self.block.append_operation(op)
        return res

    def create_program_id(self, axis: int) -> Value:
        res = Value(self.new_value_name(), I32)
        op = Operation(OpType.PROGRAM_ID, operands=[], results=[res], attributes={"axis": axis})
        if self.block:
            self.block.append_operation(op)
        return res

    def create_arange(self, start: int, end: int) -> Value:
        size = end - start
        res_type = TensorType((size,), I32)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.ARANGE, operands=[], results=[res], attributes={"start": start, "end": end})
        if self.block:
            self.block.append_operation(op)
        return res

    def create_load(self, pointer: Value, offsets: Value, mask: Optional[Value] = None) -> Value:
        ptr_type = pointer.type
        if not isinstance(ptr_type, PointerType):
            raise TypeError(f"Load source must be pointer type, got {ptr_type}")
        
        elem_type = ptr_type.element_type
        if isinstance(offsets.type, TensorType):
            res_type = TensorType(offsets.type.shape, elem_type)
        else:
            res_type = elem_type
            
        res = Value(self.new_value_name(), res_type)
        operands = [pointer, offsets]
        if mask is not None:
            operands.append(mask)
        op = Operation(OpType.LOAD, operands=operands, results=[res], attributes={"has_mask": mask is not None})
        if self.block:
            self.block.append_operation(op)
        return res

    def create_store(self, pointer: Value, offsets: Value, value: Value, mask: Optional[Value] = None) -> None:
        operands = [pointer, offsets, value]
        if mask is not None:
            operands.append(mask)
        op = Operation(OpType.STORE, operands=operands, results=[], attributes={"has_mask": mask is not None})
        if self.block:
            self.block.append_operation(op)

    def create_where(self, condition: Value, true_val: Value, false_val: Value) -> Value:
        res_type = promote_types(true_val.type, false_val.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.WHERE, operands=[condition, true_val, false_val], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_return(self, value: Optional[Value] = None) -> None:
        operands = [value] if value is not None else []
        op = Operation(OpType.RETURN, operands=operands, results=[])
        if self.block:
            self.block.append_operation(op)
