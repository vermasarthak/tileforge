"""GPU Backend IR Verifier."""

from __future__ import annotations
from tileforge.backend.metal.ir import GPUModule, GPUFunction, GPUBlock, GPUOperation, GPUOpType, AddressSpace
from tileforge.frontend.errors import IRVerificationError
from tileforge.ir.types import I1, I32, TensorType, PointerType


class GPUIRVerifier:
    """Verifies backend IR structural invariants and type correctness before code generation."""

    def verify_module(self, module: GPUModule) -> None:
        for func in module.functions:
            self.verify_function(func)

    def verify_function(self, func: GPUFunction) -> None:
        if not func.blocks:
            raise IRVerificationError(f"GPU Function @{func.name} has no blocks")

        valid_blocks = set(func.blocks)

        for block in func.blocks:
            if not block.operations:
                raise IRVerificationError(f"GPU Block ^{block.name} in @{func.name} has no operations")
            if not block.operations[-1].is_terminator():
                raise IRVerificationError(f"GPU Block ^{block.name} must end with a terminator operation")

            for op in block.operations:
                self._verify_operation(op, func, valid_blocks)

    def _verify_operation(self, op: GPUOperation, func: GPUFunction, valid_blocks: set) -> None:
        valid_op_types = set(GPUOpType)
        if op.op_type not in valid_op_types:
            raise IRVerificationError(f"Unsupported or malformed GPU operation: {op.op_type}")

        for succ in op.successors:
            if succ not in valid_blocks:
                raise IRVerificationError(f"Operation {op.op_type} references invalid branch target block ^{succ.name}")

        if op.op_type == GPUOpType.COND_BR:
            if len(op.operands) < 1:
                raise IRVerificationError("gpu.cond_br requires condition operand")
            cond = op.operands[0]
            if cond.type != I1:
                raise IRVerificationError(f"gpu.cond_br condition must be i1, got {cond.type}")
            if len(op.successors) != 2:
                raise IRVerificationError("gpu.cond_br requires exactly 2 successor blocks")

        elif op.op_type == GPUOpType.BR:
            if len(op.successors) != 1:
                raise IRVerificationError("gpu.br requires exactly 1 successor block")

        elif op.op_type == GPUOpType.GLOBAL_LOAD:
            if len(op.operands) < 2:
                raise IRVerificationError("gpu.global_load requires pointer and offset operands")
            ptr = op.operands[0]
            if not isinstance(ptr.type, PointerType):
                raise IRVerificationError(f"gpu.global_load pointer must be PointerType, got {ptr.type}")

        elif op.op_type == GPUOpType.GLOBAL_STORE:
            if len(op.operands) < 3:
                raise IRVerificationError("gpu.global_store requires pointer, offset, and value operands")
            ptr = op.operands[0]
            if not isinstance(ptr.type, PointerType):
                raise IRVerificationError(f"gpu.global_store pointer must be PointerType, got {ptr.type}")
