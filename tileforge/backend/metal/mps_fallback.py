"""Apple MPS Graph / Accelerate Fallback Partitioner.

Partitions compiler execution graph into:
1. Native Metal Tiled GEMM subgraphs (compiled directly to MSL GPU threadgroup kernels).
2. Unsupported fallback operations (routed to Apple MPS Graph / PyTorch MPS / NumPy BLAS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tileforge.ir.function import Function
from tileforge.ir.operation import Operation, OpType


@dataclass
class PartitionSubgraph:
    """Represents a partitioned subgraph for execution."""
    subgraph_id: str
    target: str  # "metal_native" or "mps_fallback"
    ops: List[Operation]


class MPSGraphPartitioner:
    """Automated Graph Partitioner splitting workloads between Metal Native & MPS Fallback."""

    def __init__(self, supported_metal_ops: Optional[List[OpType]] = None):
        if supported_metal_ops is None:
            self.supported_metal_ops = {
                OpType.PROGRAM_ID,
                OpType.CONSTANT,
                OpType.ARANGE,
                OpType.ADD,
                OpType.SUB,
                OpType.MUL,
                OpType.DIV,
                OpType.CMP,
                OpType.WHERE,
                OpType.LOAD,
                OpType.STORE,
                OpType.DOT,
                OpType.REDUCE_SUM,
                OpType.REDUCE_MAX,
                OpType.BR,
                OpType.COND_BR,
                OpType.RETURN,
            }
        else:
            self.supported_metal_ops = set(supported_metal_ops)

    def partition_function(self, func: Function) -> Tuple[List[PartitionSubgraph], Dict[str, Any]]:
        """Partitions function operations into native Metal subgraphs and MPS fallback subgraphs."""
        subgraphs: List[PartitionSubgraph] = []
        current_target = "metal_native"
        current_ops: List[Operation] = []
        subgraph_counter = 0

        for block in func.blocks:
            for op in block.operations:
                target = "metal_native" if op.op_type in self.supported_metal_ops else "mps_fallback"
                if target != current_target and current_ops:
                    subgraph_counter += 1
                    subgraphs.append(
                        PartitionSubgraph(
                            subgraph_id=f"subgraph_{subgraph_counter}",
                            target=current_target,
                            ops=list(current_ops),
                        )
                    )
                    current_ops.clear()
                    current_target = target
                current_ops.append(op)

        if current_ops:
            subgraph_counter += 1
            subgraphs.append(
                PartitionSubgraph(
                    subgraph_id=f"subgraph_{subgraph_counter}",
                    target=current_target,
                    ops=current_ops,
                )
            )

        metadata = {
            "total_subgraphs": len(subgraphs),
            "native_metal_count": sum(1 for s in subgraphs if s.target == "metal_native"),
            "mps_fallback_count": sum(1 for s in subgraphs if s.target == "mps_fallback"),
        }
        return subgraphs, metadata

    def execute_fallback_op(self, op_type: str, operands: List[np.ndarray]) -> np.ndarray:
        """Executes fallback operation via Accelerate BLAS / NumPy fallback."""
        if op_type == "matmul":
            return np.matmul(operands[0], operands[1])
        elif op_type == "exp":
            return np.exp(operands[0])
        elif op_type == "log":
            return np.log(operands[0])
        elif op_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-operands[0]))
        elif op_type == "tanh":
            return np.tanh(operands[0])
        elif op_type == "relu":
            return np.maximum(0, operands[0])
        elif op_type == "gelu":
            x = operands[0]
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        elif op_type == "softmax":
            x = operands[0]
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)
        else:
            raise NotImplementedError(f"Unsupported fallback operation '{op_type}'")
