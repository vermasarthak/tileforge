# TileForge Optimization Passes

TileForge provides an extensible IR pass framework managed by `PassManager`.

## Available Passes

### 1. Constant Fold Pass (`ConstantFoldPass`)
Folds compile-time literal computations into static constants:
- `3 + 4 -> 7`
- `5 * 8 -> 40`
- `10 < 20 -> True`

### 2. Algebraic Simplification Pass (`AlgebraicSimplifyPass`)
Applies mathematical identities to simplify arithmetic IR:
- `x + 0 -> x`
- `0 + x -> x`
- `x - 0 -> x`
- `x * 1 -> x`
- `1 * x -> x`
- `x * 0 -> 0`
- `x / 1 -> x`

### 3. Common Subexpression Elimination (`CSEPass`)
Eliminates redundant pure computations within basic blocks by reusing existing values:
```text
%0 = tf.add %a, %b
%1 = tf.add %a, %b   -->  (eliminated, uses of %1 replaced with %0)
```

### 4. Dead Code Elimination (`DeadCodeEliminationPass`)
Removes unused pure instructions whose results are consumed nowhere and which carry no side effects.
Ops like `tf.store` and `tf.return` are strictly preserved.
