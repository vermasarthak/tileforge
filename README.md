# TileForge

TileForge is an original blocked tensor-kernel compiler with a Python frontend, typed SSA IR, CFG/dominance analysis, optimization passes, reductions, tiled matrix operations, CPU reference interpreter, and native Apple Metal GPU backend.

## Overview

TileForge translates Python-like tensor kernel definitions into custom Static Single Assignment (SSA) Control Flow Graphs (CFG), computes dominance analysis, verifies IR invariants, applies optimization passes (CFG simplification, constant folding, algebraic simplification, CSE, dead-code elimination), lowers high-level IR into backend GPU IR / Metal Shading Language (MSL), and executes kernels natively on Apple Silicon GPUs (M1) or via a CPU reference interpreter.

## Attributions & Inspirations

TileForge is an independent research implementation inspired by studying:
- Blocked GPU programming models and Triton
- SSA-based compiler infrastructure (LLVM/MLIR concepts)
- Dynamic type systems and tensor compiler frontends

TileForge is built entirely from scratch and does not use or copy Triton source code or internal compiler dependencies.

## Architecture Pipeline

```text
Python Kernel (@tf.kernel)
  ↓
Parser (ast.parse + TileForge Syntax Validator)
  ↓
TileForge AST
  ↓
Semantic Analyzer (Type & Symbol Checker)
  ↓
SSA Lowering (ast_to_ir)
  ↓
TileForge Multi-Block SSA IR
  ↓
Dominance & CFG Analysis (DominanceInfo)
  ↓
Pass Manager (SimplifyCFG, Constant Folding, CSE, DCE, Algebraic Simplification)
  ↓
Optimized TileForge IR
  ↓
Target Backend Lowering (CPU Interpreter or Apple Metal GPU Backend)
```

## Backend Targets

- **CPU Interpreter**: Executes high-level TileForge SSA IR sequentially using NumPy.
- **Apple Metal GPU Backend**: Lowers high-level SSA IR to backend GPU IR and deterministically generates C++14 Metal Shading Language (MSL). Executes natively on M1 GPU using Objective-C runtime `ctypes` FFI bridge.

### Current Backend Capabilities & Limitations

- **Vector-lane lowering**: Maps `tf.arange(0, 256)` directly to threadgroup lanes (`thread_position_in_threadgroup`).
- **Control flow**: CFG branches (`tf.br`, `tf.cond_br`) and loops (`tf.range`) are translated into C++ switch-state machines to avoid MSL label constraints.
- **Reductions**: `tf.sum` and `tf.max` lower to threadgroup shared-memory parallel tree reduction primitives (`threadgroup_barrier`). Block-local reductions require multi-stage launches for global array aggregation.
- **GEMM**: `tf.dot` lowers to scalarized multiply-accumulate across block lanes.
- **Supported Types**: `f32`, `i32`, `i1`, `i64`, `ptr<T>`.

## Quick Start

```python
import tileforge as tf
from tileforge.driver import Compiler
from tileforge.ir.types import PointerType, F32, I32

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
result = compiler.compile(add, [PointerType(F32), PointerType(F32), PointerType(F32), I32])
```
