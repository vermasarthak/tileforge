"""Algebraic Simplification pass for TileForge IR."""

from __future__ import annotations

from typing import Any, Optional

from tileforge.ir.module import Module
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.value import Value
from tileforge.passes.manager import Pass


class AlgebraicSimplifyPass(Pass):
    """Applies algebraic identity simplifications to arithmetic operations."""
    def run(self, module: Module) -> bool:
        modified = False
        for func in module.functions:
            for block in func.blocks:
                i = 0
                while i < len(block.operations):
                    op = block.operations[i]
                    replacement_val = self._try_simplify_op(op)
                    if replacement_val is not None and op.results:
                        res = op.results[0]
                        res.replace_all_uses_with(replacement_val)
                        op.erase()
                        modified = True
                        continue
                    i += 1
        return modified

    def _try_simplify_op(self, op: Operation) -> Optional[Value]:
        if op.op_type == OpType.ADD:
            c0 = self._get_constant_val(op.operands[0])
            c1 = self._get_constant_val(op.operands[1])
            # x + 0 -> x
            if c1 == 0:
                return op.operands[0]
            # 0 + x -> x
            if c0 == 0:
                return op.operands[1]

        elif op.op_type == OpType.SUB:
            c1 = self._get_constant_val(op.operands[1])
            # x - 0 -> x
            if c1 == 0:
                return op.operands[0]

        elif op.op_type == OpType.MUL:
            c0 = self._get_constant_val(op.operands[0])
            c1 = self._get_constant_val(op.operands[1])
            # x * 1 -> x
            if c1 == 1:
                return op.operands[0]
            # 1 * x -> x
            if c0 == 1:
                return op.operands[1]
            # x * 0 -> 0 (return the constant 0 operand value)
            if c1 == 0:
                return op.operands[1]
            if c0 == 0:
                return op.operands[0]

        elif op.op_type == OpType.DIV:
            c1 = self._get_constant_val(op.operands[1])
            # x / 1 -> x
            if c1 == 1:
                return op.operands[0]

        return None

    def _get_constant_val(self, val: Value) -> Optional[Any]:
        if val.defining_op is not None and val.defining_op.op_type == OpType.CONSTANT:
            return val.defining_op.attributes.get("value")
        return None
