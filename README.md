# TileForge

TileForge is an experimental blocked tensor compiler with a Python frontend, typed SSA IR, CFG/dominance analysis, optimization passes, reductions, tiled matrix operations, and a CPU reference backend.

> [!NOTE]
> **Current backend:** CPU reference interpreter (NumPy)  
> **Future backend:** Lower-level code generation / GPU lowering

## Overview

TileForge translates Python-like tensor kernel definitions into custom Static Single Assignment (SSA) Control Flow Graphs (CFG), computes dominance analysis, verifies IR invariants, applies optimization passes (CFG simplification, constant folding, algebraic simplification, CSE, dead-code elimination), and executes kernels via a CPU reference interpreter.

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
CPU Reference Interpreter (NumPy Execution)
```

## Quick Start

```python
import tileforge as tf

@tf.kernel
def add(x, y, out, n):
    pid = tf.program_id(0)
    offsets = pid * 256 + tf.arange(0, 256)
    mask = offsets < n
    a = tf.load(x, offsets, mask)
    b = tf.load(y, offsets, mask)
    c = a + b
    tf.store(out, offsets, c, mask)
```
