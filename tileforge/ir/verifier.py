"""Rigorous IR Verifier for TileForge SSA IR and CFG invariant checking."""

from __future__ import annotations

from typing import Set

from tileforge.frontend.errors import IRVerificationError
from tileforge.ir.dominance import DominanceInfo
from tileforge.ir.function import Function
from tileforge.ir.module import Module
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.types import I1, I32, PointerType, TensorType
from tileforge.ir.value import Value


class IRVerifier:
    """Verifies structural validity, SSA single-definition, CFG correctness, and dominance rules of TileForge IR."""
    def verify_module(self, module: Module) -> None:
        for func in module.functions:
            if func.parent_module is not module and func.parent_module is not None:
                raise IRVerificationError(f"Function {func.name} parent module link mismatch")
            self.verify_function(func)

    def verify_function(self, func: Function) -> None:
        if not func.blocks:
            raise IRVerificationError(f"Function {func.name} has no basic blocks")

        # 1. Collect all defined values across function args & block args & op results
        defined_values: Set[Value] = set(func.args)
        value_names: Set[str] = {arg.name for arg in func.args}

        # Check function argument unique names
        if len(value_names) != len(func.args):
            raise IRVerificationError(f"Function {func.name} has duplicate argument names")

        # Verify block ownership and block arguments
        block_names: Set[str] = set()
        for block in func.blocks:
            if block.parent_function is not func and block.parent_function is not None:
                raise IRVerificationError(f"Block {block.name} parent function link mismatch")
            if block.name in block_names:
                raise IRVerificationError(f"Duplicate block label ^{block.name} in function {func.name}")
            block_names.add(block.name)

            for b_arg in block.args:
                if b_arg in defined_values:
                    raise IRVerificationError(f"Block argument {b_arg.name} defined more than once")
                if b_arg.name in value_names:
                    raise IRVerificationError(f"Block argument name {b_arg.name} is not unique")
                defined_values.add(b_arg)
                value_names.add(b_arg.name)

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

                # Verify specific operation contracts
                self._verify_operation_contracts(op, func)

        # 2. CFG Dominance Verification
        dom_info = DominanceInfo(func)
        for block in func.blocks:
            for op in block.operations:
                for operand in op.operands:
                    if not dom_info.dominates_value_use(operand, op):
                        raise IRVerificationError(
                            f"SSA Value {operand.name} used in block ^{block.name} is not dominated by its definition"
                        )

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

        elif op.op_type in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.LOGICAL_AND, OpType.LOGICAL_OR}:
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

        elif op.op_type == OpType.BR:
            if len(op.results) != 0:
                raise IRVerificationError("tf.br must produce 0 result values")
            if len(op.successors) != 1:
                raise IRVerificationError("tf.br must have exactly 1 successor block")
            target_block = op.successors[0]
            if target_block.parent_function is not func and target_block.parent_function is not None:
                raise IRVerificationError(f"Branch target ^{target_block.name} belongs to different function")
            if len(op.operands) != len(target_block.args):
                raise IRVerificationError(
                    f"Branch argument count {len(op.operands)} mismatch with target ^{target_block.name} argument count {len(target_block.args)}"
                )
            for idx, (opnd, b_arg) in enumerate(zip(op.operands, target_block.args)):
                if opnd.type != b_arg.type:
                    raise IRVerificationError(
                        f"Branch argument {idx} type {opnd.type} mismatch with target ^{target_block.name} argument {b_arg.type}"
                    )

        elif op.op_type == OpType.COND_BR:
            if len(op.results) != 0:
                raise IRVerificationError("tf.cond_br must produce 0 result values")
            if len(op.successors) != 2:
                raise IRVerificationError("tf.cond_br must have exactly 2 successor blocks (then, else)")
            cond = op.operands[0]
            if cond.type != I1:
                raise IRVerificationError(f"tf.cond_br condition operand must be i1, got {cond.type}")

            then_block = op.successors[0]
            else_block = op.successors[1]
            if then_block.parent_function is not func or else_block.parent_function is not func:
                raise IRVerificationError("Branch targets belong to different function")

            t_count = op.attributes.get("then_arg_count", 0)
            e_count = op.attributes.get("else_arg_count", 0)

            t_opnds = op.operands[1:1 + t_count]
            e_opnds = op.operands[1 + t_count:1 + t_count + e_count]

            if len(t_opnds) != len(then_block.args):
                raise IRVerificationError(f"cond_br then-args count mismatch with ^{then_block.name}")
            if len(e_opnds) != len(else_block.args):
                raise IRVerificationError(f"cond_br else-args count mismatch with ^{else_block.name}")

            for idx, (opnd, b_arg) in enumerate(zip(t_opnds, then_block.args)):
                if opnd.type != b_arg.type:
                    raise IRVerificationError(f"then branch arg {idx} type mismatch with ^{then_block.name}")
            for idx, (opnd, b_arg) in enumerate(zip(e_opnds, else_block.args)):
                if opnd.type != b_arg.type:
                    raise IRVerificationError(f"else branch arg {idx} type mismatch with ^{else_block.name}")

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
