# TileForge Implementation Plan

This document outlines the step-by-step implementation milestones for building the **TileForge** tensor-kernel compiler from scratch.

## Milestone 1: Core Setup & Documentation
- Setup repository structure (`tileforge/`, `tests/`, `examples/`, `docs/`).
- Create project config (`pyproject.toml`, `.gitignore`, `README.md`, `LICENSE`).
- Commit initial setup.

## Milestone 2: Type System & IR Core (`tileforge/ir/`)
- Implement `types.py` (`ScalarType`, `PointerType`, `TensorType`, `VoidType`).
- Implement `value.py` (`Value`, `Use`).
- Implement `operation.py` (`Operation`, opcode definitions, attributes).
- Implement `block.py`, `function.py`, `module.py`.
- Implement `builder.py` (IR construction helpers).
- Implement `printer.py` (Deterministic SSA IR text printer).
- Implement `verifier.py` (IR validity, SSA single-definition, type checking).
- Unit tests for IR core and verifier.

## Milestone 3: Compiler Frontend & Semantic Analysis (`tileforge/frontend/`, `tileforge/language/`)
- Implement `language/decorators.py` and `builtins.py` (`@tf.kernel`, builtin symbol stubs).
- Implement `frontend/ast_nodes.py` (Custom AST node hierarchy with source locations).
- Implement `frontend/symbols.py` (Lexical symbol table & scope management).
- Implement `frontend/parser.py` (Python AST to TileForge AST with strict validation & custom exceptions).
- Implement `frontend/semantic.py` (Type inference, shape checking, builtin argument checking).
- Unit tests for frontend parsing, syntax errors, symbol table, and type inference.

## Milestone 4: SSA Lowering (`tileforge/lowering/`)
- Implement `lowering/ast_to_ir.py` (TileForge AST -> SSA IR conversion).
- Support variable rebindings mapping to new SSA values.
- Unit tests for lowering vector add and masked expressions.

## Milestone 5: Optimization Passes (`tileforge/passes/`)
- Implement `manager.py` (`Pass`, `PassManager` with IR verification).
- Implement `constant_fold.py` (Scalar and block constant folding).
- Implement `algebraic.py` (Identity optimizations: `x + 0`, `x * 1`, `x * 0`, `x - 0`).
- Implement `cse.py` (Pure operation common subexpression elimination).
- Implement `dce.py` (Unused pure operation removal).
- Unit tests for each optimization pass and pass manager.

## Milestone 6: CPU Reference Interpreter & Runtime (`tileforge/runtime/`)
- Implement `interpreter.py` (NumPy memory buffer executor for IR).
- Implement grid dispatch loop for `program_id`.
- Support masked loads (with default fill) and masked stores.
- Unit tests for interpreter correctness across block sizes and unaligned lengths.

## Milestone 7: Compiler Driver & CLI (`tileforge/driver.py`, CLI)
- Implement high-level `Compiler` driver class (`compile()`, `launch()`).
- Implement CLI tool (`tileforge compile`, `tileforge run`).

## Milestone 8: Documentation, Examples & Verification
- Complete documentation (`IR.md`, `TYPE_SYSTEM.md`, `SSA.md`, `OPTIMIZATIONS.md`, `README.md`).
- Implement `examples/vector_add.py`, `examples/masked_add.py`, `examples/reduction.py`.
- Run full test suite and vector add verification against NumPy.
- Final commit checkpoint.
