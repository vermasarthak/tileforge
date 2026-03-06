# Multi-Block Control Flow Graphs (CFG) & Block-Argument SSA

TileForge implements Static Single Assignment (SSA) form across basic blocks using **block arguments** (equivalent to MLIR / Swift SIL block arguments) instead of mutable variables or explicit phi nodes.

## Branch Operations

### 1. Unconditional Branch (`tf.br`)
Transfers control unconditionally to a target basic block, passing arguments matching the target block's argument types:
```text
tf.br ^merge(%x, %y)
```

### 2. Conditional Branch (`tf.cond_br`)
Transfers control conditionally based on an `i1` boolean value to either a `then` target block or an `else` target block:
```text
tf.cond_br %cond, ^then(%a), ^else(%b)
```

## Control Flow Constructs

### Structured `if / else`
Source code:
```python
if cond:
    x = a + b
else:
    x = a - b
y = x * 2
```

IR representation:
```text
^entry:
  tf.cond_br %cond, ^then, ^else

^then:
  %0 = tf.add %a, %b : f32
  tf.br ^merge(%0)

^else:
  %1 = tf.sub %a, %b : f32
  tf.br ^merge(%1)

^merge(%x: f32):
  %2 = tf.mul %x, %c2 : f32
  tf.return %2
```

### Counted Range Loops (`tf.range`)
Source code:
```python
acc = 0.0
for i in tf.range(0, N):
    acc = acc + x[i]
```

IR representation:
```text
^entry:
  %zero = tf.constant 0 : i32
  %acc0 = tf.constant 0.0 : f32
  tf.br ^loop_header(%zero, %acc0)

^loop_header(%i: i32, %acc: f32):
  %cond = tf.cmp lt %i, %N : i1
  tf.cond_br %cond, ^loop_body, ^loop_exit(%acc)

^loop_body:
  %val = tf.load %x[%i] : f32
  %acc_next = tf.add %acc, %val : f32
  %one = tf.constant 1 : i32
  %i_next = tf.add %i, %one : i32
  tf.br ^loop_header(%i_next, %acc_next)

^loop_exit(%result: f32):
  tf.return %result
```
