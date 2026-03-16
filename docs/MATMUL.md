# Tiled Matrix Multiplication (`tf.dot`) in TileForge

TileForge provides explicit compiler operations and typing rules for blocked matrix multiplication tiles.

## `tf.dot` Operation

```text
%C = tf.dot %A, %B : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
```

### Type Contract & Constraints
- Operands `%A` and `%B` must be 2D `TensorType` instances (`tensor<MxKxT>` and `tensor<KxNxT>`).
- Inner contraction dimensions must match (`K == K`).
- Output tensor type is `tensor<MxNxT>`.
- Type checking verifies dimension compatibility statically during semantic analysis.

## Tiled Matmul Kernel Pattern

```python
@tf.kernel
def matmul_kernel(A, B, C, M, N, K):
    pid_m = tf.program_id(0)
    pid_n = tf.program_id(1)

    BLOCK_M = 16
    BLOCK_N = 16

    offs_m = pid_m * BLOCK_M + tf.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tf.arange(0, BLOCK_N)

    acc = tf.zeros((16, 16), "f32")

    for k in tf.range(0, K):
        a = tf.load(A, offs_m)
        b = tf.load(B, offs_n)
        acc = acc + tf.dot(a, b)

    tf.store(C, offs_m, acc)
```
