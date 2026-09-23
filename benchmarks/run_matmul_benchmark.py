"""Tiled GEMM Benchmark Suite for TileForge Apple Metal GPU vs NumPy CPU."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tileforge as tf
from tileforge.backend.metal.runtime import (
    MetalBuffer,
    MetalDevice,
    MTLSize,
    get_sel,
    libobjc,
    msg_send_dispatch,
    msg_send_id,
    msg_send_set_buffer,
    msg_send_set_bytes,
    msg_send_void,
    msg_send_void_id,
)
from tileforge.driver import Compiler
from tileforge.ir.types import F32, I32, PointerType

msg_send_double = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", libobjc))

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

def benchmark_matmul(M: int, N: int, K: int, trials: int = 50, warmups: int = 10):
    compiler = Compiler(optimize=True, backend="metal")
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32, I32, I32]
    res = compiler.compile(matmul_kernel, arg_types)

    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    # 1. NumPy CPU execution timing
    for _ in range(warmups):
        A_np @ B_np

    cpu_times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        A_np @ B_np
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1000.0)

    # 2. End-to-end Metal launch timing
    grid_m = (M + 15) // 16
    grid_n = (N + 15) // 16
    grid = (grid_n, grid_m)

    for _ in range(warmups):
        res.launch(grid=grid, args=[A_np, B_np, C_np, M, N, K])

    e2e_times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        res.launch(grid=grid, args=[A_np, B_np, C_np, M, N, K])
        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000.0)

    # 3. Steady-state Metal GPU kernel-only timing using persistent buffers
    device = res.compiled_kernel.runtime.device
    pipeline = res.compiled_kernel.pipeline_state
    cmd_queue = device.command_queue

    buf_A = MetalBuffer(device, M * K * 4)
    buf_B = MetalBuffer(device, K * N * 4)
    buf_C = MetalBuffer(device, M * N * 4)

    buf_A.upload_numpy(A_np)
    buf_B.upload_numpy(B_np)

    c_M = ctypes.c_int32(M)
    c_N = ctypes.c_int32(N)
    c_K = ctypes.c_int32(K)

    grid_size = MTLSize(grid[0], grid[1], 1)
    tg_size = MTLSize(16, 16, 1)

    gpu_kernel_times = []
    for _ in range(warmups + trials):
        cmd_buf = msg_send_id(cmd_queue, get_sel("commandBuffer"))
        encoder = msg_send_id(cmd_buf, get_sel("computeCommandEncoder"))
        msg_send_void_id(encoder, get_sel("setComputePipelineState:"), pipeline)
        msg_send_set_buffer(encoder, get_sel("setBuffer:offset:atIndex:"), buf_A.buf, 0, 0)
        msg_send_set_buffer(encoder, get_sel("setBuffer:offset:atIndex:"), buf_B.buf, 0, 1)
        msg_send_set_buffer(encoder, get_sel("setBuffer:offset:atIndex:"), buf_C.buf, 0, 2)
        msg_send_set_bytes(encoder, get_sel("setBytes:length:atIndex:"), ctypes.byref(c_M), 4, 3)
        msg_send_set_bytes(encoder, get_sel("setBytes:length:atIndex:"), ctypes.byref(c_N), 4, 4)
        msg_send_set_bytes(encoder, get_sel("setBytes:length:atIndex:"), ctypes.byref(c_K), 4, 5)

        msg_send_dispatch(encoder, get_sel("dispatchThreadgroups:threadsPerThreadgroup:"), grid_size, tg_size)
        msg_send_void(encoder, get_sel("endEncoding"))
        msg_send_void(cmd_buf, get_sel("commit"))
        msg_send_void(cmd_buf, get_sel("waitUntilCompleted"))

        t_start = msg_send_double(cmd_buf, get_sel("GPUStartTime"))
        t_end = msg_send_double(cmd_buf, get_sel("GPUEndTime"))
        gpu_kernel_times.append((t_end - t_start) * 1000.0)

    gpu_kernel_times = gpu_kernel_times[warmups:]

    def stats(arr):
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "stddev_ms": float(np.std(arr)),
        }

    return {
        "M": M, "N": N, "K": K,
        "trials": trials,
        "warmups": warmups,
        "numpy_cpu_compute": stats(cpu_times),
        "metal_end_to_end": stats(e2e_times),
        "metal_gpu_kernel_only": stats(gpu_kernel_times),
    }

def run_suite():
    os.makedirs("benchmarks/results", exist_ok=True)
    dev = MetalDevice()
    print(f"Running Tiled GEMM Benchmark Suite on {dev.get_name()}...")

    results = []
    for dim in [64, 128, 256, 512]:
        print(f"  Measuring {dim}x{dim} @ {dim}x{dim}...")
        res = benchmark_matmul(dim, dim, dim, trials=50, warmups=10)
        results.append(res)
        print(f"    NumPy CPU: {res['numpy_cpu_compute']['median_ms']:.4f} ms | Metal GPU Kernel: {res['metal_gpu_kernel_only']['median_ms']:.4f} ms | Metal E2E: {res['metal_end_to_end']['median_ms']:.4f} ms")

    out_file = "benchmarks/results/tiled_matmul_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    run_suite()
