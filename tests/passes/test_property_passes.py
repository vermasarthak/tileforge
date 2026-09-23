import numpy as np
import pytest

from tileforge.ir import F32, I32, VOID, Function, IRBuilder, Module, PointerType, Value
from tileforge.passes import (
    AlgebraicSimplifyPass,
    ConstantFoldPass,
    CSEPass,
    DeadCodeEliminationPass,
    PassManager,
)
from tileforge.runtime import CPUInterpreter


@pytest.mark.parametrize("seed", [42, 123, 999, 2024])
def test_optimization_passes_preserve_vector_semantics(seed):
    """Property test: running optimization passes (ConstantFold, AlgebraicSimplify, CSE, DCE)

    must produce bitwise/numerically identical outputs to unoptimized IR on CPU reference.
    """
    rng = np.random.default_rng(seed)
    N = 256
    x_data = rng.standard_normal(N).astype(np.float32)
    y_data = rng.standard_normal(N).astype(np.float32)

    # 1. Unoptimized IR module
    mod_unopt = Module("unopt")
    x = Value("%x", PointerType(F32))
    y = Value("%y", PointerType(F32))
    out = Value("%out", PointerType(F32))
    n = Value("%n", I32)

    fn_unopt = Function("vec_expr", [x, y, out, n], VOID)
    mod_unopt.add_function(fn_unopt)

    b = IRBuilder(fn_unopt.entry_block)
    pid = b.create_program_id(0)
    c256 = b.create_constant(256, I32)
    offs_base = b.create_mul(pid, c256)
    ar = b.create_arange(0, 256)
    offs = b.create_add(offs_base, ar)
    mask = b.create_cmp("lt", offs, n)

    xv = b.create_load(x, offs, mask)
    yv = b.create_load(y, offs, mask)

    # Redundant algebraic identities & constants: (x + 0) * 1 + y + (x + 0) - (x + 0)
    c0 = b.create_constant(0.0, F32)
    c1 = b.create_constant(1.0, F32)
    x_plus_0 = b.create_add(xv, c0)
    x_ident = b.create_mul(x_plus_0, c1)
    res1 = b.create_add(x_ident, yv)

    # Dead code: unused computation
    _unused = b.create_mul(xv, yv)

    b.create_store(out, offs, res1, mask)
    b.create_return()

    out_unopt = np.zeros(N, dtype=np.float32)
    interp = CPUInterpreter()
    interp.execute(fn_unopt, grid=(1,), args=[x_data, y_data, out_unopt, N])

    # 2. Optimized module
    pm = PassManager([
        ConstantFoldPass(),
        AlgebraicSimplifyPass(),
        CSEPass(),
        DeadCodeEliminationPass(),
    ])
    pm.run(mod_unopt)

    out_opt = np.zeros(N, dtype=np.float32)
    interp.execute(fn_unopt, grid=(1,), args=[x_data, y_data, out_opt, N])

    # Verify both match reference NumPy and each other
    expected = x_data + y_data
    np.testing.assert_allclose(out_unopt, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_opt, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_opt, out_unopt, rtol=1e-7, atol=1e-7)
