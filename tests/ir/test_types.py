import pytest
from tileforge.ir.types import (
    I1, I32, F32, PointerType, TensorType, VOID,
    promote_types, compare_types
)
from tileforge.frontend.errors import TypeCheckError


def test_primitive_types():
    assert str(I32) == "i32"
    assert I32.is_integer()
    assert not I32.is_float()
    assert F32.is_float()
    assert I32.bit_width() == 32


def test_pointer_type():
    ptr = PointerType(F32)
    assert str(ptr) == "ptr<f32>"
    assert ptr.is_pointer()


def test_tensor_type():
    tensor = TensorType((256,), I32)
    assert str(tensor) == "tensor<256xi32>"
    assert tensor.is_tensor()


def test_promote_types():
    assert promote_types(I32, I32) == I32
    assert promote_types(I32, F32) == F32
    
    t256_f32 = TensorType((256,), F32)
    assert promote_types(t256_f32, F32) == t256_f32
    assert promote_types(t256_f32, t256_f32) == t256_f32

    t128_f32 = TensorType((128,), F32)
    with pytest.raises(TypeCheckError):
        promote_types(t256_f32, t128_f32)


def test_compare_types():
    assert compare_types(I32, I32) == I1
    t256_f32 = TensorType((256,), F32)
    assert compare_types(t256_f32, t256_f32) == TensorType((256,), I1)
