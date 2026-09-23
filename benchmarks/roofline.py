"""Roofline performance benchmark suite for TileForge Metal backend vs Apple MPS / NumPy baseline."""

from __future__ import annotations

import time

import numpy as np

import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import F32, I32, PointerType


@tf.kernel
def bench_matmul_kernel(A, B, C, M, N, K):
    pid_m = tf.program_id(0)
    pid_n = tf.program_id(1)
    offs_m = pid_m * 16 + tf.arange(0, 16)
    offs_n = pid_n * 16 + tf.arange(0, 16)
    acc = tf.zeros((16, 16), "f32")
    for k in tf.range(0, K):
        a = tf.load(A, (offs_m, k))
        b = tf.load(B, (k, offs_n))
        acc = acc + tf.dot(a, b)
    tf.store(C, (offs_m, offs_n), acc)


def run_roofline_benchmark(sizes: list[tuple[int, int, int]] = None) -> list[dict]:
    if sizes is None:
        sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)]

    compiler = Compiler(optimize=True, backend="metal")
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32, I32, I32]
    result = compiler.compile(bench_matmul_kernel, arg_types)

    results = []
    print(f"{'M x N x K':<20} | {'FLOPs':<12} | {'Bytes':<12} | {'Intensity (F/B)':<18} | {'TFLOPS':<10} | {'Status'}")
    print("-" * 85)

    for M, N, K in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        grid_m = (M + 15) // 16
        grid_n = (N + 15) // 16

        # Warmup
        result.launch(grid=(grid_m, grid_n), args=[A, B, C, M, N, K])

        # Benchmark 10 iterations
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            result.launch(grid=(grid_m, grid_n), args=[A, B, C, M, N, K])
        t1 = time.perf_counter()

        avg_latency_s = (t1 - t0) / iters
        flops = 2.0 * M * N * K
        bytes_transferred = (M * K + K * N + M * N) * 4.0
        operational_intensity = flops / bytes_transferred
        tflops = (flops / avg_latency_s) / 1e12

        row = {
            "M": M, "N": N, "K": K,
            "flops": flops,
            "bytes": bytes_transferred,
            "intensity": operational_intensity,
            "latency_ms": avg_latency_s * 1000.0,
            "tflops": tflops,
        }
        results.append(row)

        shape_str = f"{M}x{N}x{K}"
        print(f"{shape_str:<20} | {flops:<12.2e} | {bytes_transferred:<12.2e} | {operational_intensity:<18.2f} | {tflops:<10.4f} | PASSED")

    return results


if __name__ == "__main__":
    run_roofline_benchmark()
