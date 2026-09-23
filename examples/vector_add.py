"""Vector addition example in TileForge."""

import numpy as np

import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import F32, I32, PointerType


@tf.kernel
def add(x, y, out, n):
    pid = tf.program_id(0)
    offsets = pid * 256 + tf.arange(0, 256)
    mask = offsets < n
    a = tf.load(x, offsets, mask)
    b = tf.load(y, offsets, mask)
    c = a + b
    tf.store(out, offsets, c, mask)


def run_vector_add_demo():
    print("Compiling Vector Add Kernel...")
    compiler = Compiler(optimize=True, backend="metal")
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32]

    result = compiler.compile(add, arg_types)

    print("\n--- IR BEFORE OPTIMIZATION ---")
    print(result.ir_before_optimization)

    print("\n--- IR AFTER OPTIMIZATION ---")
    print(result.ir_after_optimization)

    test_lengths = [1, 17, 255, 256, 257, 1000, 4097]
    block_size = 256

    print("\nVerifying Correctness across test lengths...")
    for N in test_lengths:
        x_np = np.random.randn(N).astype(np.float32)
        y_np = np.random.randn(N).astype(np.float32)
        out_np = np.zeros(N, dtype=np.float32)

        expected_np = x_np + y_np

        grid_x = (N + block_size - 1) // block_size
        result.launch(grid=(grid_x,), args=[x_np, y_np, out_np, N])

        np.testing.assert_allclose(out_np, expected_np, rtol=1e-5, atol=1e-5)
        print(f"  ✓ N={N:4d}: Correctness verified against NumPy!")


if __name__ == "__main__":
    run_vector_add_demo()
