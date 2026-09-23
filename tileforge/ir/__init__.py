"""TileForge SSA IR module."""

from tileforge.ir.block import Block
from tileforge.ir.builder import IRBuilder
from tileforge.ir.function import Function
from tileforge.ir.module import Module
from tileforge.ir.operation import PURE_OPERATIONS, Operation, OpType
from tileforge.ir.printer import IRPrinter
from tileforge.ir.types import (
    BF16,
    F16,
    F32,
    F64,
    I1,
    I8,
    I16,
    I32,
    I64,
    VOID,
    PointerType,
    PrimitiveType,
    TensorType,
    Type,
    VoidType,
    compare_types,
    promote_types,
)
from tileforge.ir.value import Use, Value
from tileforge.ir.verifier import IRVerifier

__all__ = [
    "Type",
    "PrimitiveType",
    "PointerType",
    "TensorType",
    "VoidType",
    "I1",
    "I8",
    "I16",
    "I32",
    "I64",
    "F16",
    "BF16",
    "F32",
    "F64",
    "VOID",
    "promote_types",
    "compare_types",
    "Value",
    "Use",
    "Operation",
    "OpType",
    "PURE_OPERATIONS",
    "Block",
    "Function",
    "Module",
    "IRBuilder",
    "IRPrinter",
    "IRVerifier",
]
