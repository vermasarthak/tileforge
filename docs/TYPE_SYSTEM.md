# TileForge Type System

TileForge enforces strict, deterministic static typing across scalar, pointer, tensor block, and void types.

## Type Categories

### 1. Primitive Scalar Types
- Integers: `i1` (boolean), `i8`, `i16`, `i32`, `i64`
- Floating Point: `f16`, `bf16`, `f32`, `f64`

### 2. Pointer Types
- `ptr<T>`: Points to contiguous memory buffers of element type `T`.
- Example: `ptr<f32>`, `ptr<i32>`.

### 3. Tensor Block Types
- `tensor<shape x element_type>`: Represents multidimensional blocked tensor tiles.
- Examples: `tensor<256xf32>`, `tensor<32x32xi32>`, `tensor<256xi1>`.

### 4. Void Type
- `void`: Represents operations/functions producing no values.

## Promotion and Type Rules

1. **Scalar + Scalar**: Promotes to highest bit-width matching primitive category.
2. **Tensor + Tensor**: Shapes must match exactly. Element types promote deterministically.
3. **Tensor + Scalar**: Scalar is promoted to match tensor element type and shape.
4. **Comparisons**: Produce `i1` for scalar operands or `tensor<shape x i1>` for tensor operands.
5. **No Implicit Arbitrary Broadcasting**: TileForge explicitly prohibits implicit multi-dimensional broadcasting beyond scalar-to-tensor promotion.
