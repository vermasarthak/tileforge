"""Constant Folding optimization pass for TileForge IR."""

from __future__ import annotations

from typing import Any, Optional

from tileforge.ir.builder import IRBuilder
from tileforge.ir.module import Module
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.value import Value
from tileforge.passes.manager import Pass


class ConstantFoldPass(Pass):
    """Folds operations with compile-time constant operands into constant values."""
    def run(self, module: Module) -> bool:
        modified = False
        for func in module.functions:
            for block in func.blocks:
                builder = IRBuilder(block)
                # Iterate ops in block
                i = 0
                while i < len(block.operations):
                    op = block.operations[i]
                    folded_val = self._try_fold_op(op, builder)
                    if folded_val is not None and op.results:
                        res = op.results[0]
                        res.replace_all_uses_with(folded_val)
                        op.erase()
                        modified = True
                        # Do not increment i because op was erased
                        continue
                    i += 1
        return modified

    def _try_fold_op(self, op: Operation, builder: IRBuilder) -> Optional[Value]:
        if op.op_type in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}:
            c0 = self._get_constant_val(op.operands[0])
            c1 = self._get_constant_val(op.operands[1])
            if c0 is not None and c1 is not None:
                res_type = op.results[0].type
                if op.op_type == OpType.ADD:
                    val = c0 + c1
                elif op.op_type == OpType.SUB:
                    val = c0 - c1
                elif op.op_type == OpType.MUL:
                    val = c0 * c1
                elif op.op_type == OpType.DIV:
                    if c1 == 0 or c1 == 0.0:
                        return None
                    val = c0 // c1 if isinstance(c0, int) and isinstance(c1, int) else c0 / c1
                else:
                    return None

                # Create constant before op
                builder._value_counter = int(op.results[0].name.lstrip("%")) if op.results[0].name.lstrip("%").isdigit() else builder._value_counter
                new_const = builder.create_constant(val, res_type)
                # Move created op before target op
                created_op = builder.block.operations.pop()
                idx = builder.block.operations.index(op)
                builder.block.operations.insert(idx, created_op)
                return new_const

        elif op.op_type == OpType.CMP:
            c0 = self._get_constant_val(op.operands[0])
            c1 = self._get_constant_val(op.operands[1])
            if c0 is not None and c1 is not None:
                pred = op.attributes.get("predicate")
                res_type = op.results[0].type
                res_bool = False
                if pred == "lt":
                    res_bool = c0 < c1
                elif pred == "le":
                    res_bool = c0 <= c1
                elif pred == "gt":
                    res_bool = c0 > c1
                elif pred == "ge":
                    res_bool = c0 >= c1
                elif pred == "eq":
                    res_bool = c0 == c1
                elif pred == "ne":
                    res_bool = c0 != c1

                new_const = builder.create_constant(res_bool, res_type)
                created_op = builder.block.operations.pop()
                idx = builder.block.operations.index(op)
                builder.block.operations.insert(idx, created_op)
                return new_const

        return None

    def _get_constant_val(self, val: Value) -> Optional[Any]:
        if val.defining_op is not None and val.defining_op.op_type == OpType.CONSTANT:
            return val.defining_op.attributes.get("value")
        return None
