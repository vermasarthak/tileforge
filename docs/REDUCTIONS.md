# Reduction Operations in TileForge

TileForge supports built-in block reduction operations.

## Operations & Semantics

### 1. `tf.sum` (`tf.reduce_sum`)
Computes the sum of elements across a tensor tile:
```text
%sum = tf.reduce_sum %x : tensor<256xf32> -> f32
```
- **Input:** 1D or 2D tensor block.
- **Output:** Scalar element type or reduced dimension.
- **Empty Reduction Behavior:** Returns additive identity (`0` or `0.0`).

### 2. `tf.max` (`tf.reduce_max`)
Computes the maximum value across a tensor tile:
```text
%max = tf.reduce_max %x : tensor<256xf32> -> f32
```
- **Input:** 1D or 2D tensor block.
- **Output:** Scalar element type.
- **Empty Reduction Behavior:** Returns minimum primitive bound.
