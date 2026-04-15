"""GPU Backend Intermediate Representation (GPU IR) for TileForge.

Represents GPU execution concepts explicitly:
- Address spaces (device global memory vs threadgroup local memory)
- Execution IDs (grid threadgroup id, thread position in threadgroup, block size)
- Memory allocation (threadgroup shared memory allocation)
- Lane predicates and scalarized operations
- Memory barriers and threadgroup synchronization
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Union
from tileforge.ir.types import Type, PrimitiveType, PointerType, TensorType, VOID, I32, I1, F32, I64


class AddressSpace(Enum):
    GLOBAL = "device"
    THREADGROUP = "threadgroup"
    CONSTANT = "constant"
    PRIVATE = "thread"


class GPUOpType(Enum):
    # Execution indexing
    PROGRAM_ID = "gpu.program_id"
    THREAD_ID = "gpu.thread_id"
    THREAD_ID_X = "gpu.thread_id_x"
    THREAD_ID_Y = "gpu.thread_id_y"
    THREADGROUP_ID_X = "gpu.threadgroup_id_x"
    THREADGROUP_ID_Y = "gpu.threadgroup_id_y"
    BLOCK_DIM = "gpu.block_dim"

    # Constants & Literals
    CONSTANT = "gpu.constant"

    # Scalar & Vector arithmetic
    ADD = "gpu.add"
    SUB = "gpu.sub"
    MUL = "gpu.mul"
    DIV = "gpu.div"
    CMP = "gpu.cmp"

    # Logical
    LOGICAL_AND = "gpu.logical_and"
    LOGICAL_OR = "gpu.logical_or"
    SELECT = "gpu.select"

    # Memory Access
    GLOBAL_LOAD = "gpu.global_load"
    GLOBAL_STORE = "gpu.global_store"
    LOCAL_ALLOC = "gpu.local_alloc"
    LOCAL_LOAD = "gpu.local_load"
    LOCAL_STORE = "gpu.local_store"

    # Synchronizations
    BARRIER = "gpu.barrier"

    # Reductions & Matrix ops
    REDUCE_SUM = "gpu.reduce_sum"
    REDUCE_MAX = "gpu.reduce_max"
    DOT = "gpu.dot"
    TILED_DOT = "gpu.tiled_dot"

    # Control flow
    BR = "gpu.br"
    COND_BR = "gpu.cond_br"
    RETURN = "gpu.return"


class GPUValue:
    """Represents a value/variable in GPU IR."""
    def __init__(self, name: str, value_type: Type, address_space: Optional[AddressSpace] = None):
        self.name: str = name
        self.type: Type = value_type
        self.address_space: Optional[AddressSpace] = address_space
        self.defining_op: Optional[GPUOperation] = None

    def __repr__(self) -> str:
        addr_str = f" {self.address_space.value}" if self.address_space else ""
        return f"%{self.name}: {self.type}{addr_str}"


class GPUOperation:
    """Represents an instruction in GPU IR."""
    def __init__(
        self,
        op_type: GPUOpType,
        operands: Optional[List[GPUValue]] = None,
        results: Optional[List[GPUValue]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        successors: Optional[List[GPUBlock]] = None,
    ):
        self.op_type: GPUOpType = op_type
        self.operands: List[GPUValue] = operands if operands is not None else []
        self.results: List[GPUValue] = results if results is not None else []
        self.attributes: Dict[str, Any] = attributes if attributes is not None else {}
        self.successors: List[GPUBlock] = successors if successors is not None else []
        self.parent_block: Optional[GPUBlock] = None

        for res in self.results:
            res.defining_op = self

    def is_terminator(self) -> bool:
        return self.op_type in {GPUOpType.BR, GPUOpType.COND_BR, GPUOpType.RETURN}

    def __repr__(self) -> str:
        res_str = ", ".join(r.name for r in self.results) + " = " if self.results else ""
        opnd_str = ", ".join(o.name for o in self.operands)
        attr_str = f" {self.attributes}" if self.attributes else ""
        succ_str = f" [{', '.join('^' + s.name for s in self.successors)}]" if self.successors else ""
        return f"{res_str}{self.op_type.value} {opnd_str}{attr_str}{succ_str}"


class GPUBlock:
    """Basic block in GPU IR containing linear operations."""
    def __init__(self, name: str, parent_function: Optional[GPUFunction] = None):
        self.name: str = name
        self.parent_function: Optional[GPUFunction] = parent_function
        self.args: List[GPUValue] = []
        self.operations: List[GPUOperation] = []

    def append_operation(self, op: GPUOperation) -> None:
        op.parent_block = self
        self.operations.append(op)

    @property
    def terminator(self) -> Optional[GPUOperation]:
        if self.operations and self.operations[-1].is_terminator():
            return self.operations[-1]
        return None

    def __repr__(self) -> str:
        args_str = f"({', '.join(a.name + ': ' + str(a.type) for a in self.args)})" if self.args else ""
        return f"^{self.name}{args_str}:"


class GPUKernelArg:
    """Kernel parameter definition with ABI information."""
    def __init__(self, name: str, arg_type: Type, address_space: AddressSpace = AddressSpace.GLOBAL):
        self.name: str = name
        self.type: Type = arg_type
        self.address_space: AddressSpace = address_space

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type} ({self.address_space.value})"


class GPUFunction:
    """GPU Kernel function containing basic blocks and ABI argument signature."""
    def __init__(self, name: str, args: List[GPUKernelArg]):
        self.name: str = name
        self.args: List[GPUKernelArg] = args
        self.blocks: List[GPUBlock] = []
        self.is_2d_grid: bool = False

    def create_block(self, name: str) -> GPUBlock:
        block = GPUBlock(name, parent_function=self)
        self.blocks.append(block)
        return block

    @property
    def entry_block(self) -> GPUBlock:
        if not self.blocks:
            raise ValueError(f"GPU Function '{self.name}' has no basic blocks")
        return self.blocks[0]


class GPUModule:
    """Module containing GPU functions."""
    def __init__(self):
        self.functions: List[GPUFunction] = []

    def add_function(self, func: GPUFunction) -> None:
        self.functions.append(func)
