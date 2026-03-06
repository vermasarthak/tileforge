"""IRBuilder for constructing TileForge SSA IR operations and control flow."""

from __future__ import annotations
from typing import List, Optional, Any, Tuple
from tileforge.ir.types import (
    Type, PrimitiveType, TensorType, PointerType, I1, I32, F32, VOID,
    promote_types, compare_types
)
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

    def create_block_arg(self, block: Block, type_: Type, name_prefix: str = "arg") -> Value:
        val_name = self.new_value_name()
        val = Value(val_name, type_)
        block.add_argument(val)
        return val

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

    def create_logical_and(self, lhs: Value, rhs: Value) -> Value:
        res_type = promote_types(lhs.type, rhs.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.LOGICAL_AND, operands=[lhs, rhs], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_logical_or(self, lhs: Value, rhs: Value) -> Value:
        res_type = promote_types(lhs.type, rhs.type)
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.LOGICAL_OR, operands=[lhs, rhs], results=[res])
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

    def create_load(self, pointer: Value, offsets: Union[Value, Tuple[Value, ...], List[Value]], mask: Optional[Value] = None) -> Value:
        ptr_type = pointer.type
        if not isinstance(ptr_type, PointerType):
            raise TypeError(f"Load source must be pointer type, got {ptr_type}")
        
        elem_type = ptr_type.element_type
        
        offset_list = list(offsets) if isinstance(offsets, (tuple, list)) else [offsets]
        tensor_shapes = [v.type.shape for v in offset_list if isinstance(v.type, TensorType)]
        
        if tensor_shapes:
            res_type = TensorType(tensor_shapes[0], elem_type)
        else:
            res_type = elem_type
            
        res = Value(self.new_value_name(), res_type)
        operands = [pointer] + offset_list
        if mask is not None:
            operands.append(mask)
        op = Operation(OpType.LOAD, operands=operands, results=[res], attributes={"has_mask": mask is not None})
        if self.block:
            self.block.append_operation(op)
        return res

    def create_store(self, pointer: Value, offsets: Union[Value, Tuple[Value, ...], List[Value]], value: Value, mask: Optional[Value] = None) -> None:
        offset_list = list(offsets) if isinstance(offsets, (tuple, list)) else [offsets]
        operands = [pointer] + offset_list + [value]
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

    def create_br(self, target_block: Block, dest_args: Optional[List[Value]] = None) -> None:
        args = dest_args if dest_args is not None else []
        op = Operation(OpType.BR, operands=args, results=[], successors=[target_block])
        if self.block:
            self.block.append_operation(op)

    def create_cond_br(
        self,
        condition: Value,
        then_block: Block,
        else_block: Block,
        then_args: Optional[List[Value]] = None,
        else_args: Optional[List[Value]] = None,
    ) -> None:
        t_args = then_args if then_args is not None else []
        e_args = else_args if else_args is not None else []
        
        # Track split argument count in attributes
        operands = [condition] + t_args + e_args
        attrs = {"then_arg_count": len(t_args), "else_arg_count": len(e_args)}
        
        op = Operation(
            OpType.COND_BR,
            operands=operands,
            results=[],
            attributes=attrs,
            successors=[then_block, else_block],
        )
        if self.block:
            self.block.append_operation(op)

    def create_reduce_sum(self, tensor: Value) -> Value:
        if not isinstance(tensor.type, TensorType):
            raise TypeError(f"reduce_sum operand must be tensor type, got {tensor.type}")
        res_type = tensor.type.element_type
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.REDUCE_SUM, operands=[tensor], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_reduce_max(self, tensor: Value) -> Value:
        if not isinstance(tensor.type, TensorType):
            raise TypeError(f"reduce_max operand must be tensor type, got {tensor.type}")
        res_type = tensor.type.element_type
        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.REDUCE_MAX, operands=[tensor], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_dot(self, lhs: Value, rhs: Value) -> Value:
        if not isinstance(lhs.type, TensorType) or not isinstance(rhs.type, TensorType):
            raise TypeError(f"dot operands must be TensorType, got {lhs.type} and {rhs.type}")
        
        if len(lhs.type.shape) == 1 and len(rhs.type.shape) == 1:
            res_type = TensorType((lhs.type.shape[0], rhs.type.shape[0]), lhs.type.element_type)
        elif len(lhs.type.shape) == 2 and len(rhs.type.shape) == 2:
            m, k1 = lhs.type.shape
            k2, n = rhs.type.shape
            if k1 != k2:
                raise ValueError(f"dot inner dimension mismatch: {k1} vs {k2}")
            res_type = TensorType((m, n), lhs.type.element_type)
        else:
            res_type = TensorType((lhs.type.shape[0], rhs.type.shape[-1]), lhs.type.element_type)

        res = Value(self.new_value_name(), res_type)
        op = Operation(OpType.DOT, operands=[lhs, rhs], results=[res])
        if self.block:
            self.block.append_operation(op)
        return res

    def create_return(self, value: Optional[Value] = None) -> None:
        operands = [value] if value is not None else []
        op = Operation(OpType.RETURN, operands=operands, results=[])
        if self.block:
            self.block.append_operation(op)
