"""TileForge SSA IR module."""

from tileforge.ir.types import (
    Type,
    PrimitiveType,
    PointerType,
    TensorType,
    VoidType,
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
    VOID,
    promote_types,
    compare_types,
)
from tileforge.ir.value import Value, Use
from tileforge.ir.operation import Operation, OpType, PURE_OPERATIONS
from tileforge.ir.block import Block
from tileforge.ir.function import Function
from tileforge.ir.module import Module
from tileforge.ir.builder import IRBuilder
from tileforge.ir.printer import IRPrinter
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
