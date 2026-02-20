"""Rigorous IR Verifier for TileForge SSA IR invariant checking."""

from __future__ import annotations
from typing import Set
from tileforge.frontend.errors import IRVerificationError
from tileforge.ir.module import Module
from tileforge.ir.function import Function
from tileforge.ir.block import Block
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.value import Value
from tileforge.ir.types import PointerType, TensorType, VoidType, I1, I32


class IRVerifier:
    """Verifies structural validity, SSA single-definition, and typing rules of TileForge IR."""
    def verify_module(self, module: Module) -> None:
        for func in module.functions:
            if func.parent_module is not module and func.parent_module is not None:
                raise IRVerificationError(f"Function {func.name} parent module link mismatch")
            self.verify_function(func)

    def verify_function(self, func: Function) -> None:
        defined_values: Set[Value] = set(func.args)
        value_names: Set[str] = {arg.name for arg in func.args}

        # Check function argument unique names
        if len(value_names) != len(func.args):
            raise IRVerificationError(f"Function {func.name} has duplicate argument names")

        for block in func.blocks:
            if block.parent_function is not func and block.parent_function is not None:
                raise IRVerificationError(f"Block {block.name} parent function link mismatch")

            if not block.operations:
                raise IRVerificationError(f"Block {block.name} in function {func.name} is empty")

            # Terminator check: last op must be terminator
            last_op = block.operations[-1]
            if not last_op.is_terminator():
                raise IRVerificationError(f"Block {block.name} last operation is not a valid terminator")

            for op_idx, op in enumerate(block.operations):
                if op.parent_block is not block and op.parent_block is not None:
                    raise IRVerificationError(f"Operation {op.op_type} parent block link mismatch")

                # Non-last operation should not be terminator
                if op.is_terminator() and op_idx != len(block.operations) - 1:
                    raise IRVerificationError(f"Terminator operation {op.op_type} found before block end")

                # Verify operands are defined prior to use
                for operand in op.operands:
                    if operand not in defined_values:
                        raise IRVerificationError(
                            f"Operation {op.op_type} uses undefined operand {operand.name}"
                        )

                # Verify result SSA single-definition
                for res in op.results:
                    if res in defined_values:
                        raise IRVerificationError(f"SSA Value {res.name} defined more than once")
                    if res.name in value_names:
                        raise IRVerificationError(f"SSA Value name {res.name} is not unique")
                    if res.defining_op is not op:
                        raise IRVerificationError(f"Result {res.name} defining_op reference invalid")

                    defined_values.add(res)
                    value_names.add(res.name)

                # Verify specific operation invariants
                self._verify_operation_contracts(op, func)

    def _verify_operation_contracts(self, op: Operation, func: Function) -> None:
        if op.op_type == OpType.CONSTANT:
            if len(op.results) != 1 or len(op.operands) != 0:
                raise IRVerificationError("tf.constant must have 0 operands and 1 result")
            if "value" not in op.attributes:
                raise IRVerificationError("tf.constant requires 'value' attribute")

        elif op.op_type == OpType.PROGRAM_ID:
            if len(op.results) != 1 or len(op.operands) != 0:
                raise IRVerificationError("tf.program_id must have 0 operands and 1 result")
            if op.results[0].type != I32:
                raise IRVerificationError("tf.program_id result type must be i32")
            if "axis" not in op.attributes:
                raise IRVerificationError("tf.program_id requires 'axis' attribute")

        elif op.op_type == OpType.ARANGE:
            if len(op.results) != 1 or len(op.operands) != 0:
                raise IRVerificationError("tf.arange must have 0 operands and 1 result")
            if not isinstance(op.results[0].type, TensorType) or op.results[0].type.element_type != I32:
                raise IRVerificationError("tf.arange result type must be tensor<...xi32>")
            if "start" not in op.attributes or "end" not in op.attributes:
                raise IRVerificationError("tf.arange requires 'start' and 'end' attributes")

        elif op.op_type in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}:
            if len(op.results) != 1 or len(op.operands) != 2:
                raise IRVerificationError(f"{op.op_type} must have 2 operands and 1 result")

        elif op.op_type == OpType.CMP:
            if len(op.results) != 1 or len(op.operands) != 2:
                raise IRVerificationError("tf.cmp must have 2 operands and 1 result")
            if "predicate" not in op.attributes:
                raise IRVerificationError("tf.cmp requires 'predicate' attribute")

        elif op.op_type == OpType.LOAD:
            if len(op.results) != 1 or len(op.operands) not in (2, 3):
                raise IRVerificationError("tf.load must have 2 or 3 operands (ptr, offsets[, mask]) and 1 result")
            if not isinstance(op.operands[0].type, PointerType):
                raise IRVerificationError("tf.load first operand must be a PointerType")

        elif op.op_type == OpType.STORE:
            if len(op.results) != 0:
                raise IRVerificationError("tf.store must produce no result values")
            if len(op.operands) not in (3, 4):
                raise IRVerificationError("tf.store must have 3 or 4 operands (ptr, offsets, val[, mask])")
            if not isinstance(op.operands[0].type, PointerType):
                raise IRVerificationError("tf.store first operand must be a PointerType")

        elif op.op_type == OpType.RETURN:
            if len(op.results) != 0:
                raise IRVerificationError("tf.return must produce 0 result values")
            if func.return_type.is_void():
                if len(op.operands) != 0:
                    raise IRVerificationError("Void function return operation cannot take operands")
            else:
                if len(op.operands) != 1:
                    raise IRVerificationError(f"Non-void function return operation must take 1 operand, got {len(op.operands)}")
                if op.operands[0].type != func.return_type:
                    raise IRVerificationError(f"Return operand type {op.operands[0].type} mismatch with function return type {func.return_type}")
