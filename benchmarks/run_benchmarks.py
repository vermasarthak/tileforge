"""Reproducible Benchmark Suite for TileForge Apple Metal GPU vs NumPy CPU."""

from __future__ import annotations
import time
import os
import sys
import json
import platform
import ctypes
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import PointerType, F32, I32
from tileforge.backend.metal.runtime import (
    MetalDevice,
    MetalBuffer,
    msg_send_id,
    msg_send_void,
    msg_send_void_id,
    msg_send_set_buffer,
    msg_send_set_bytes,
    msg_send_dispatch,
    get_sel,
    MTLSize,
    libobjc,
)

msg_send_double = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", libobjc))

def benchmark_vector_add(N: int = 1_000_000, trials: int = 100, warmups: int = 10):
    @tf.kernel
    def add(x, y, out, n):
        pid = tf.program_id(0)
        offsets = pid * 256 + tf.arange(0, 256)
        mask = offsets < n
        a = tf.load(x, offsets, mask)
        b = tf.load(y, offsets, mask)
        c = a + b
        tf.store(out, offsets, c, mask)

    compiler = Compiler(optimize=True, backend="metal")
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32]
    res = compiler.compile(add, arg_types)

    x_np = np.random.randn(N).astype(np.float32)
    y_np = np.random.randn(N).astype(np.float32)
    out_np = np.zeros(N, dtype=np.float32)

    # 1. NumPy CPU execution timing
    for _ in range(warmups):
        np_res = x_np + y_np

    cpu_times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        np_res = x_np + y_np
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1000.0)

    # 2. End-to-end Metal launch timing
    grid = ((N + 255) // 256,)
    for _ in range(warmups):
        res.launch(grid=grid, args=[x_np, y_np, out_np, N])

    e2e_times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        res.launch(grid=grid, args=[x_np, y_np, out_np, N])
        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000.0)

    # 3. Steady-state Metal kernel-only & transfer breakdown timing using persistent buffers
    device = res.compiled_kernel.runtime.device
    pipeline = res.compiled_kernel.pipeline_state
    cmd_queue = device.command_queue

    t0_alloc = time.perf_counter()
    buf_x = MetalBuffer(device, N * 4)
    buf_y = MetalBuffer(device, N * 4)
    buf_out = MetalBuffer(device, N * 4)
    t1_alloc = time.perf_counter()
    alloc_time = (t1_alloc - t0_alloc) * 1000.0

    t0_up = time.perf_counter()
    buf_x.upload_numpy(x_np)
    buf_y.upload_numpy(y_np)
    t1_up = time.perf_counter()
    upload_time = (t1_up - t0_up) * 1000.0

    c_n = ctypes.c_int32(N)
    grid_size = MTLSize(grid[0], 1, 1)
    tg_size = MTLSize(256, 1, 1)

    gpu_kernel_times = []
    for _ in range(warmups + trials):
        cmd_buf = msg_send_id(cmd_queue, get_sel("commandBuffer"))
        encoder = msg_send_id(cmd_buf, get_sel("computeCommandEncoder"))
        msg_send_void_id(encoder, get_sel("setComputePipelineState:"), pipeline)
        msg_send_set_buffer(encoder, get_sel("setBuffer:offset:atIndex:"), buf_x.buf, 0, 0)
        msg_send_set_buffer(encoder, get_sel("setBuffer:offset:atIndex:"), buf_y.buf, 0, 1)
        msg_send_set_buffer(encoder, get_sel("setBuffer:offset:atIndex:"), buf_out.buf, 0, 2)
        msg_send_set_bytes(encoder, get_sel("setBytes:length:atIndex:"), ctypes.byref(c_n), 4, 3)

        msg_send_dispatch(encoder, get_sel("dispatchThreadgroups:threadsPerThreadgroup:"), grid_size, tg_size)
        msg_send_void(encoder, get_sel("endEncoding"))
        msg_send_void(cmd_buf, get_sel("commit"))
        msg_send_void(cmd_buf, get_sel("waitUntilCompleted"))

        t_start = msg_send_double(cmd_buf, get_sel("GPUStartTime"))
        t_end = msg_send_double(cmd_buf, get_sel("GPUEndTime"))
        gpu_kernel_times.append((t_end - t_start) * 1000.0)

    gpu_kernel_times = gpu_kernel_times[warmups:]

    t0_dn = time.perf_counter()
    buf_out.download_numpy(out_np)
    t1_dn = time.perf_counter()
    download_time = (t1_dn - t0_dn) * 1000.0

    def stats(arr):
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "stddev_ms": float(np.std(arr)),
        }

    return {
        "N": N,
        "trials": trials,
        "warmups": warmups,
        "numpy_cpu_compute": stats(cpu_times),
        "metal_end_to_end": stats(e2e_times),
        "metal_gpu_kernel_only": stats(gpu_kernel_times),
        "breakdown_ms": {
            "buffer_allocation": alloc_time,
            "host_to_device_upload": upload_time,
            "device_to_host_download": download_time,
        }
    }

def run_suite():
    os.makedirs("benchmarks/results", exist_ok=True)
    dev = MetalDevice()
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.machine(),
        "chip": dev.get_name(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
    }

    results = {"metadata": meta, "vector_add": []}

    print("Running Vector Add Benchmark Suite...")
    for n in [1_000, 16_000, 256_000, 1_000_000]:
        print(f"  Measuring N={n}...")
        res = benchmark_vector_add(N=n, trials=50, warmups=10)
        results["vector_add"].append(res)

    out_file = "benchmarks/results/metal_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    run_suite()
