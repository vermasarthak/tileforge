# Static Single Assignment (SSA) in TileForge

In Static Single Assignment (SSA) form, every variable is assigned exactly once, and every variable use is defined before its usage.

## Variable Re-assignment & Versioning

In source code, variables are frequently re-assigned:

```python
x = a + b
x = x * 2
```

In TileForge IR, the lowering pass maps each assignment target to a **new SSA value** rather than mutating existing instructions:

```text
%0 = tf.add %a, %b : f32
%1 = tf.constant 2 : i32
%2 = tf.mul %0, %1 : f32
```

Symbol table bindings map `x` dynamically during AST traversal:
- After statement 1: `env['x'] -> %0`
- During statement 2: `x` evaluates to `%0`, result produces `%2`, `env['x'] -> %2`

## Invariants Verified by IR Verifier

1. **Definition Uniqueness**: No SSA value name/identity is defined more than once in a module/function.
2. **Dominance / Pre-definition**: Every operand used in an operation must be defined prior to its use (either as a function argument or by an earlier operation in the control-flow block).
3. **Use Tracking**: Every SSA `Value` maintains an accurate list of `Use` objects representing consuming operations.
