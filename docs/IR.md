# TileForge Intermediate Representation (IR)

TileForge IR is an explicit, strongly typed Static Single Assignment (SSA) intermediate representation for tensor-kernel compilation.

## Core IR Concepts

### Value
A `Value` represents a typed SSA result produced by an operation or passed as a function parameter.
Each `Value` is immutable and defined exactly once.
- `name`: Human-readable SSA identifier (e.g., `%0`, `%x`).
- `type`: Explicit TileForge `Type`.
- `defining_op`: Reference to defining `Operation` (or `None` for function arguments).
- `uses`: List of `Use` objects tracking operations consuming this value.

### Operation
An `Operation` represents a single IR instruction.
- `op_type`: Opcode identifier (e.g. `tf.add`, `tf.load`, `tf.constant`).
- `operands`: List of input `Value` references.
- `results`: List of output `Value` references (0 for side-effecting ops like `tf.store`).
- `attributes`: Static compile-time metadata dictionary (e.g. `axis`, `value`, `predicate`).
- `parent_block`: Reference to containing `Block`.

### Block
A basic block containing a sequential sequence of `Operation` instances ending in a terminator instruction (`tf.return`).

### Function
A kernel container containing argument definitions, return type, and basic blocks.

### Module
Top-level compilation unit containing function definitions.

## Standard Opcodes

| Opcode | Operands | Results | Description |
|---|---|---|---|
| `tf.constant` | None | `%res` | Emits literal scalar value |
| `tf.program_id` | None | `%res` | Emits current launch program ID along specified axis |
| `tf.arange` | None | `%res` | Emits contiguous integer vector range `[start, end)` |
| `tf.add` | `%lhs, %rhs` | `%res` | Elementwise addition |
| `tf.sub` | `%lhs, %rhs` | `%res` | Elementwise subtraction |
| `tf.mul` | `%lhs, %rhs` | `%res` | Elementwise multiplication |
| `tf.div` | `%lhs, %rhs` | `%res` | Elementwise division |
| `tf.cmp` | `%lhs, %rhs` | `%res` | Comparison (`lt`, `le`, `gt`, `ge`, `eq`, `ne`) |
| `tf.load` | `%ptr, %offs[, %mask]` | `%res` | Masked pointer load from buffer |
| `tf.store` | `%ptr, %offs, %val[, %mask]` | None | Masked store into buffer |
| `tf.where` | `%cond, %true_val, %false_val` | `%res` | Ternary conditional selection |
| `tf.return` | `[%val]` | None | Returns control from kernel block |
