"""Control flow example (if/else and for range loops) in TileForge."""

import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import F32, I32, PointerType


@tf.kernel
def control_flow_kernel(x, out, flag, n):
    pid = tf.program_id(0)
    offsets = pid * 256 + tf.arange(0, 256)
    mask = offsets < n
    val = tf.load(x, offsets, mask)

    if flag != 0:
        acc = 0.0
        for i in tf.range(0, 4):
            acc = acc + val
        res = acc
    else:
        res = val * 0.5

    tf.store(out, offsets, res, mask)


def main():
    print("Compiling Control Flow Kernel...")
    compiler = Compiler(optimize=True)
    arg_types = [PointerType(F32), PointerType(F32), I32, I32]

    res = compiler.compile(control_flow_kernel, arg_types)
    print("\n--- OPTIMIZED IR ---")
    print(res.ir_after_optimization)
    print("\n--- CFG BLOCKS ---")
    print(" -> ".join(res.cfg))


if __name__ == "__main__":
    main()
