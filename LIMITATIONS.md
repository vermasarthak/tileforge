# TileForge Limitations & Operational Boundaries

TileForge is an educational and experimental tensor compiler designed to explore typed SSA representation, CFG analysis, compiler optimization passes, and Metal code generation.

## Supported Capabilities

- **Front-end Grammar**: Restricted Python functions decorated with `@tf.kernel`, arithmetic expressions (`+`, `-`, `*`, `/`), control flow (`if`/`else`), indexing via `tf.program_id`, `tf.arange`, `tf.load`, and `tf.store`.
- **Supported Data Types**: `f32`, `i32`, `i64`, `bool`, and typed pointers (`PointerType(T)`).
- **Compilation Passes**:
  - `ConstantFoldPass`: Compile-time evaluation of constant scalar operations.
  - `AlgebraicSimplifyPass`: Canonical simplification (e.g. `x + 0 -> x`, `x * 1 -> x`, `x * 0 -> 0`).
  - `CSEPass`: Intra-block common subexpression elimination with Value-numbering.
  - `DeadCodeEliminationPass`: Pruning of dead/unused SSA instructions without side-effects.
  - `SimplifyCFGPass`: Constant branch folding and unreachable block pruning.
- **Execution Backends**:
  - **CPU Reference Interpreter**: NumPy-backed reference engine supporting verification across arbitrary platforms (macOS / Linux).
  - **Apple Metal GPU Runtime**: Native Metal compute pipeline execution via Objective-C runtime bridge (`ctypes`), validated on Apple Silicon (`M1`+).

## Explicit Limitations & Unsupported Features

- **No Dynamic Memory Allocation**: Dynamic memory allocation inside GPU kernels is unsupported; all buffer memory must be pre-allocated.
- **No Arbitrary Python Runtime**: Arbitrary Python loops, dynamic calls, class definitions, and object mutations inside `@tf.kernel` functions are rejected by the AST verifier.
- **Architecture Specificity**: Direct GPU kernel compilation to Metal Shading Language (`MSL`) requires macOS with Apple Silicon Metal support. On non-macOS environments (e.g. Linux CI), only the CPU reference interpreter, parser, SSA IR, and pass verifications are supported.
- **FP16 / BF16 / INT8 Support**: Currently experimental/partial; primary end-to-end verified numerical pipeline is `float32`.
- **Not a Drop-in Replacement for PyTorch/Triton**: TileForge is an experimental research system designed for learning and verifying compiler internals, not a production-grade inference engine or broad compiler runtime.
