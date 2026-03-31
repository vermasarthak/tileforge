import pytest
import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import PointerType, F32, I32

def test_vector_add_codegen_snapshot():
    @tf.kernel
    def add(x, y, out, n):
        pid = tf.program_id(0)
        offs = pid * 256 + tf.arange(0, 256)
        mask = offs < n
        a = tf.load(x, offs, mask)
        b = tf.load(y, offs, mask)
        tf.store(out, offs, a + b, mask)

    compiler = Compiler(optimize=True, backend="metal")
    res = compiler.compile(add, [PointerType(F32), PointerType(F32), PointerType(F32), I32])
    msl = res.generated_source

    assert "#include <metal_stdlib>" in msl
    assert "kernel void add(" in msl
    assert "device float* var_x [[buffer(0)]]" in msl
    assert "uint3 tid [[thread_position_in_threadgroup]]" in msl
    assert "switch (_state)" in msl

def test_no_hardcoding_multiple_kernels():
    @tf.kernel
    def k1(x, y, out, n):
        pid = tf.program_id(0)
        offs = pid * 256 + tf.arange(0, 256)
        mask = offs < n
        a = tf.load(x, offs, mask)
        b = tf.load(y, offs, mask)
        tf.store(out, offs, a + b, mask)

    @tf.kernel
    def k2(x, y, z, out, n):
        pid = tf.program_id(0)
        offs = pid * 256 + tf.arange(0, 256)
        mask = offs < n
        a = tf.load(x, offs, mask)
        b = tf.load(y, offs, mask)
        c = tf.load(z, offs, mask)
        tf.store(out, offs, a * b + c, mask)

    compiler = Compiler(optimize=True, backend="metal")
    res1 = compiler.compile(k1, [PointerType(F32), PointerType(F32), PointerType(F32), I32])
    res2 = compiler.compile(k2, [PointerType(F32), PointerType(F32), PointerType(F32), PointerType(F32), I32])

    assert res1.generated_source != res2.generated_source
    assert "buffer(3)" in res1.generated_source
    assert "buffer(4)" in res2.generated_source

def test_cfg_runtime_execution():
    @tf.kernel
    def loop_kernel(x, out, n):
        pid = tf.program_id(0)
        offs = pid * 256 + tf.arange(0, 256)
        mask = offs < n
        val = tf.load(x, offs, mask)
        acc = 0.0
        for i in tf.range(0, 5):
            acc = acc + val
        tf.store(out, offs, acc, mask)

    compiler_metal = Compiler(optimize=True, backend="metal")
    compiler_cpu = Compiler(optimize=True, backend="cpu")

    arg_types = [PointerType(F32), PointerType(F32), I32]
    res_metal = compiler_metal.compile(loop_kernel, arg_types)
    res_cpu = compiler_cpu.compile(loop_kernel, arg_types)

    import numpy as np
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out_metal = np.zeros(3, dtype=np.float32)
    out_cpu = np.zeros(3, dtype=np.float32)

    res_metal.launch(grid=(1,), args=[x, out_metal, 3])
    res_cpu.launch(grid=(1,), args=[x, out_cpu, 3])

    np.testing.assert_allclose(out_metal, out_cpu, rtol=1e-5, atol=1e-5)
