"""TileForge type system for scalars, pointers, tensors, and void."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union

from tileforge.frontend.errors import TypeCheckError


class Type:
    """Base class for all TileForge types."""
    def is_integer(self) -> bool:
        return False

    def is_float(self) -> bool:
        return False

    def is_numeric(self) -> bool:
        return self.is_integer() or self.is_float()

    def is_pointer(self) -> bool:
        return False

    def is_tensor(self) -> bool:
        return False

    def is_void(self) -> bool:
        return False


@dataclass(frozen=True)
class PrimitiveType(Type):
    name: str

    def __str__(self) -> str:
        return self.name

    def is_integer(self) -> bool:
        return self.name in {"i1", "i8", "i16", "i32", "i64"}

    def is_float(self) -> bool:
        return self.name in {"f16", "bf16", "f32", "f64"}

    def bit_width(self) -> int:
        if self.name in {"i1"}:
            return 1
        elif self.name in {"i8"}:
            return 8
        elif self.name in {"i16", "f16", "bf16"}:
            return 16
        elif self.name in {"i32", "f32"}:
            return 32
        elif self.name in {"i64", "f64"}:
            return 64
        raise ValueError(f"Unknown bit width for type {self.name}")


# Scalar Singletons
I1 = PrimitiveType("i1")
I8 = PrimitiveType("i8")
I16 = PrimitiveType("i16")
I32 = PrimitiveType("i32")
I64 = PrimitiveType("i64")

F16 = PrimitiveType("f16")
BF16 = PrimitiveType("bf16")
F32 = PrimitiveType("f32")
F64 = PrimitiveType("f64")


@dataclass(frozen=True)
class VoidType(Type):
    def __str__(self) -> str:
        return "void"

    def is_void(self) -> bool:
        return True


VOID = VoidType()


@dataclass(frozen=True)
class PointerType(Type):
    element_type: PrimitiveType

    def __post_init__(self):
        if not isinstance(self.element_type, PrimitiveType):
            raise TypeCheckError(f"Pointer element type must be primitive, got {self.element_type}")

    def __str__(self) -> str:
        return f"ptr<{self.element_type}>"

    def is_pointer(self) -> bool:
        return True


@dataclass(frozen=True)
class TensorType(Type):
    shape: Tuple[int, ...]
    element_type: PrimitiveType

    def __post_init__(self):
        if not isinstance(self.element_type, PrimitiveType):
            raise TypeCheckError(f"Tensor element type must be primitive, got {self.element_type}")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise TypeCheckError(f"Tensor shape must consist of positive integers, got {self.shape}")

    def __str__(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{dims}x{self.element_type}>"

    def is_tensor(self) -> bool:
        return True


def promote_types(lhs: Type, rhs: Type) -> Type:
    """Deterministically promote two types for binary operations.
    
    Supports:
    - Scalar + Scalar (must match or promote correctly)
    - Tensor + Tensor (exact shape & promoted element type)
    - Tensor + Scalar / Scalar + Tensor (promotes scalar to tensor element type)
    """
    if isinstance(lhs, TensorType) and isinstance(rhs, TensorType):
        if lhs.shape != rhs.shape:
            raise TypeCheckError(f"Incompatible tensor shapes for binary operation: {lhs.shape} vs {rhs.shape}")
        elem_type = promote_scalar_types(lhs.element_type, rhs.element_type)
        return TensorType(lhs.shape, elem_type)
    
    if isinstance(lhs, TensorType) and isinstance(rhs, PrimitiveType):
        elem_type = promote_scalar_types(lhs.element_type, rhs)
        return TensorType(lhs.shape, elem_type)
    
    if isinstance(lhs, PrimitiveType) and isinstance(rhs, TensorType):
        elem_type = promote_scalar_types(lhs, rhs.element_type)
        return TensorType(rhs.shape, elem_type)
    
    if lhs == rhs:
        return lhs

    if isinstance(lhs, PrimitiveType) and isinstance(rhs, PrimitiveType):
        return promote_scalar_types(lhs, rhs)

    raise TypeCheckError(f"Cannot perform binary operation between {lhs} and {rhs}")


def promote_scalar_types(lhs: PrimitiveType, rhs: PrimitiveType) -> PrimitiveType:
    if lhs == rhs:
        return lhs

    # Integer & Integer promotion
    if lhs.is_integer() and rhs.is_integer():
        # Prefer higher bit width
        return lhs if lhs.bit_width() >= rhs.bit_width() else rhs

    # Float & Float promotion
    if lhs.is_float() and rhs.is_float():
        return lhs if lhs.bit_width() >= rhs.bit_width() else rhs

    # Integer & Float promotion -> promote integer to float
    if lhs.is_integer() and rhs.is_float():
        return rhs
    if lhs.is_float() and rhs.is_integer():
        return lhs

    raise TypeCheckError(f"Incompatible scalar types: {lhs} and {rhs}")


def compare_types(lhs: Type, rhs: Type) -> Type:
    """Determine result type of comparison operation (<, <=, >, >=, ==, !=).
    
    Returns i1 for scalars, tensor<shape x i1> for tensor operands.
    """
    if isinstance(lhs, TensorType) and isinstance(rhs, TensorType):
        if lhs.shape != rhs.shape:
            raise TypeCheckError(f"Incompatible tensor shapes for comparison: {lhs.shape} vs {rhs.shape}")
        return TensorType(lhs.shape, I1)
    
    if isinstance(lhs, TensorType) and isinstance(rhs, PrimitiveType):
        return TensorType(lhs.shape, I1)
    
    if isinstance(lhs, PrimitiveType) and isinstance(rhs, TensorType):
        return TensorType(rhs.shape, I1)
    
    if isinstance(lhs, PrimitiveType) and isinstance(rhs, PrimitiveType):
        return I1

    raise TypeCheckError(f"Cannot compare {lhs} and {rhs}")
