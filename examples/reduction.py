"""Reduction operations example in TileForge (tf.sum and tf.max)."""

import numpy as np

import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import F32, I32, PointerType


@tf.kernel
def reduction_kernel(x, out_sum, out_max, n):
    pid = tf.program_id(0)
    offsets = pid * 256 + tf.arange(0, 256)
    mask = offsets < n
    val = tf.load(x, offsets, mask)

    # Neutral fill for out-of-bounds elements
    val_sum = tf.where(mask, val, 0.0)
    val_max = tf.where(mask, val, -1e9)

    sum_val = tf.sum(val_sum)
    max_val = tf.max(val_max)

    tf.store(out_sum, pid, sum_val)
    tf.store(out_max, pid, max_val)


def run_reduction_demo():
    print("Compiling Reduction Kernel (tf.sum & tf.max)...")
    compiler = Compiler(optimize=True)
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32]

    result = compiler.compile(reduction_kernel, arg_types)

    print("\n--- OPTIMIZED IR ---")
    print(result.ir_after_optimization)

    test_lengths = [1, 7, 16, 17, 255, 256, 257, 1000]

    print("\nVerifying Reductions across test lengths...")
    for N in test_lengths:
        x_np = np.random.randn(N).astype(np.float32)
        grid_x = (N + 255) // 256
        out_sum = np.zeros(grid_x, dtype=np.float32)
        out_max = np.zeros(grid_x, dtype=np.float32)

        result.launch(grid=(grid_x,), args=[x_np, out_sum, out_max, N])

        valid_len = min(N, 256)
        expected_sum = np.sum(x_np[:valid_len])
        expected_max = np.max(x_np[:valid_len])

        np.testing.assert_allclose(out_sum[0], expected_sum, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_max[0], expected_max, rtol=1e-4, atol=1e-4)
        print(f"  ✓ N={N:4d}: tf.sum & tf.max correctness verified against NumPy!")


if __name__ == "__main__":
    run_reduction_demo()
