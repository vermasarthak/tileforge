import numpy as np
import pytest
import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import PointerType, F32, I32


@tf.kernel
def add_kernel(x, y, out, n):
    pid = tf.program_id(0)
    offsets = pid * 256 + tf.arange(0, 256)
    mask = offsets < n
    a = tf.load(x, offsets, mask)
    b = tf.load(y, offsets, mask)
    c = a + b
    tf.store(out, offsets, c, mask)


@pytest.mark.parametrize("N", [1, 17, 255, 256, 257, 1000, 4097])
def test_vector_add_e2e(N):
    compiler = Compiler(optimize=True)
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32]
    
    result = compiler.compile(add_kernel, arg_types)

    x_np = np.random.randn(N).astype(np.float32)
    y_np = np.random.randn(N).astype(np.float32)
    out_np = np.zeros(N, dtype=np.float32)

    expected = x_np + y_np

    block_size = 256
    grid_x = (N + block_size - 1) // block_size

    result.launch(grid=(grid_x,), args=[x_np, y_np, out_np, N])

    np.testing.assert_allclose(out_np, expected, rtol=1e-5, atol=1e-5)
