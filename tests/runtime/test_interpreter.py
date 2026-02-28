import numpy as np
import pytest
from tileforge.ir import (
    Module, Function, Value, IRBuilder, PointerType, F32, I32, VOID
)
from tileforge.runtime import CPUInterpreter


def test_interpreter_execution():
    module = Module("test_exec")
    x = Value("%x", PointerType(F32))
    y = Value("%y", PointerType(F32))
    out = Value("%out", PointerType(F32))
    n = Value("%n", I32)

    func = Function("add", [x, y, out, n], VOID)
    module.add_function(func)

    builder = IRBuilder(func.entry_block)
    pid = builder.create_program_id(0)
    c256 = builder.create_constant(256, I32)
    step1 = builder.create_mul(pid, c256)
    ar = builder.create_arange(0, 256)
    offs = builder.create_add(step1, ar)
    mask = builder.create_cmp("lt", offs, n)
    xv = builder.create_load(x, offs, mask)
    yv = builder.create_load(y, offs, mask)
    zv = builder.create_add(xv, yv)
    builder.create_store(out, offs, zv, mask)
    builder.create_return()

    N = 300
    x_arr = np.linspace(1.0, 10.0, N, dtype=np.float32)
    y_arr = np.linspace(2.0, 20.0, N, dtype=np.float32)
    out_arr = np.zeros(N, dtype=np.float32)

    interpreter = CPUInterpreter()
    grid = (2,)  # 2 * 256 = 512 >= 300
    interpreter.execute(func, grid=grid, args=[x_arr, y_arr, out_arr, N])

    expected = x_arr + y_arr
    np.testing.assert_allclose(out_arr, expected, rtol=1e-5)
