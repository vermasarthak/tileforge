import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import F32, I32, PointerType


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

def test_regression_no_whole_function_template_routing():
    import inspect

    from tileforge.backend.metal import codegen
    source = inspect.getsource(codegen)
    assert "generate_tiled_matmul_function" not in source
    assert "has_tiled_dot" not in source

def test_custom_arg_names_and_surrounding_ops():
    @tf.kernel
    def custom_matmul(p, q, r, s, t, u):
        pid_m = tf.program_id(0)
        pid_n = tf.program_id(1)
        offs_m = pid_m * 16 + tf.arange(0, 16)
        offs_n = pid_n * 16 + tf.arange(0, 16)
        acc = tf.zeros((16, 16), "f32")
        for k in tf.range(0, u):
            a = tf.load(p, (offs_m, k))
            b = tf.load(q, (k, offs_n))
            acc = acc + tf.dot(a, b)
        tf.store(r, (offs_m, offs_n), acc)

    compiler = Compiler(optimize=True, backend="metal")
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32, I32, I32]
    res = compiler.compile(custom_matmul, arg_types)
    msl = res.generated_source

    assert "var_p" in msl
    assert "var_q" in msl
    assert "var_r" in msl
    assert "var_s" in msl
    assert "var_t" in msl
    assert "var_u" in msl
    assert "switch (_state)" in msl

def test_post_dot_surrounding_ops_e2e():
    @tf.kernel
    def fused_matmul(A, B, C, M, N, K, scale, bias):
        pid_m = tf.program_id(0)
        pid_n = tf.program_id(1)
        offs_m = pid_m * 16 + tf.arange(0, 16)
        offs_n = pid_n * 16 + tf.arange(0, 16)
        acc = tf.zeros((16, 16), "f32")
        for k in tf.range(0, K):
            a = tf.load(A, (offs_m, k))
            b = tf.load(B, (k, offs_n))
            acc = acc + tf.dot(a, b)
        tmp1 = acc * scale
        tmp2 = tmp1 + bias
        tmp3 = tmp2 * 2.0
        tf.store(C, (offs_m, offs_n), tmp3)

    compiler = Compiler(optimize=True, backend="metal")
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32, I32, I32, F32, F32]
    result = compiler.compile(fused_matmul, arg_types)

    import numpy as np
    M, N, K = 16, 16, 16
    scale, bias = 0.5, 1.5
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    dot_expected = np.matmul(A_np, B_np)
    expected = (dot_expected * scale + bias) * 2.0

    grid_m = (M + 15) // 16
    grid_n = (N + 15) // 16
    result.launch(grid=(grid_m, grid_n), args=[A_np, B_np, C_np, M, N, K, scale, bias])

    np.testing.assert_allclose(C_np, expected, rtol=1e-4, atol=1e-4)

def test_reordered_parameters_matmul_e2e():
    @tf.kernel
    def reordered_kernel(M, N, K, OutBuf, InA, InB):
        pid_m = tf.program_id(0)
        pid_n = tf.program_id(1)
        offs_m = pid_m * 16 + tf.arange(0, 16)
        offs_n = pid_n * 16 + tf.arange(0, 16)
        acc = tf.zeros((16, 16), "f32")
        for k in tf.range(0, K):
            a = tf.load(InA, (offs_m, k))
            b = tf.load(InB, (k, offs_n))
            acc = acc + tf.dot(a, b)
        tf.store(OutBuf, (offs_m, offs_n), acc)

    compiler = Compiler(optimize=True, backend="metal")
    arg_types = [I32, I32, I32, PointerType(F32), PointerType(F32), PointerType(F32)]
    result = compiler.compile(reordered_kernel, arg_types)

    import numpy as np
    M, N, K = 16, 16, 16
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    expected = np.matmul(A_np, B_np)
    grid_m = (M + 15) // 16
    grid_n = (N + 15) // 16

    result.launch(grid=(grid_m, grid_n), args=[M, N, K, C_np, A_np, B_np])
    np.testing.assert_allclose(C_np, expected, rtol=1e-4, atol=1e-4)
