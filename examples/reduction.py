"""Blocked kernel example with conditional logic in TileForge."""

import numpy as np
import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import PointerType, F32, I32


@tf.kernel
def scale_kernel(x, out, scale, n):
    pid = tf.program_id(0)
    offsets = pid * 256 + tf.arange(0, 256)
    mask = offsets < n
    val = tf.load(x, offsets, mask)
    scaled = val * scale
    tf.store(out, offsets, scaled, mask)


def main():
    print("Compiling Scale Kernel...")
    compiler = Compiler(optimize=True)
    arg_types = [PointerType(F32), PointerType(F32), F32, I32]
    res = compiler.compile(scale_kernel, arg_types)
    print("\n--- OPTIMIZED IR ---")
    print(res.ir_after_optimization)

    N = 500
    x_np = np.ones(N, dtype=np.float32) * 3.0
    out_np = np.zeros(N, dtype=np.float32)
    grid_x = (N + 255) // 256
    
    res.launch(grid=(grid_x,), args=[x_np, out_np, 2.5, N])
    np.testing.assert_allclose(out_np, 7.5, rtol=1e-5)
    print("✓ Scale Kernel execution verified!")


if __name__ == "__main__":
    main()
