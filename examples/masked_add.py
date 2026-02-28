"""Masked elementwise addition example in TileForge."""

import numpy as np
import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import PointerType, F32, I32


@tf.kernel
def masked_add(x, y, out, n):
    pid = tf.program_id(0)
    offsets = pid * 256 + tf.arange(0, 256)
    mask = offsets < n
    a = tf.load(x, offsets, mask)
    b = tf.load(y, offsets, mask)
    cond = a > 0.0
    combined_mask = mask and cond
    c = a + b
    tf.store(out, offsets, c, mask=combined_mask)


def main():
    print("Compiling Masked Add Kernel...")
    compiler = Compiler(optimize=True)
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32]
    res = compiler.compile(masked_add, arg_types)
    print("\n--- OPTIMIZED IR ---")
    print(res.ir_after_optimization)


if __name__ == "__main__":
    main()
