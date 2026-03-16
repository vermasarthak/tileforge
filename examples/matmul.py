"""Tiled matrix multiplication example in TileForge."""

import numpy as np
import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import PointerType, F32, I32


@tf.kernel
def matmul_kernel(A, B, C, M, N, K):
    pid_m = tf.program_id(0)
    pid_n = tf.program_id(1)

    BLOCK_M = 16
    BLOCK_N = 16

    offs_m = pid_m * BLOCK_M + tf.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tf.arange(0, BLOCK_N)

    acc = tf.zeros((16, 16), "f32")

    for k in tf.range(0, K):
        a = tf.load(A, (offs_m, k))
        b = tf.load(B, (k, offs_n))
        acc = acc + tf.dot(a, b)

    tf.store(C, (offs_m, offs_n), acc)


def run_matmul_demo():
    print("Compiling Tiled Matmul Kernel...")
    compiler = Compiler(optimize=True)
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32, I32, I32]

    result = compiler.compile(matmul_kernel, arg_types)

    print("\n--- OPTIMIZED IR ---")
    print(result.ir_after_optimization)

    print("\nVerifying Correctness across matrix shapes...")
    test_shapes = [
        (1, 1, 1),
        (8, 8, 8),
        (16, 16, 16),
        (17, 17, 17),
        (31, 23, 19),
        (64, 64, 64),
    ]

    for M, N, K in test_shapes:
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)
        C_np = np.zeros((M, N), dtype=np.float32)

        expected = np.matmul(A_np, B_np)

        # Launch kernel
        grid_m = (M + 15) // 16
        grid_n = (N + 15) // 16

        result.launch(grid=(grid_m, grid_n), args=[A_np, B_np, C_np, M, N, K])

        np.testing.assert_allclose(C_np, expected, rtol=1e-4, atol=1e-4)
        print(f"  ✓ Shape {M:2d}x{K:2d} @ {K:2d}x{N:2d}: Correctness verified against NumPy!")


if __name__ == "__main__":
    run_matmul_demo()
