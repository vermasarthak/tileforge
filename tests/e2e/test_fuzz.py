import random

import numpy as np

from tileforge.ir import F32, I32, VOID, Function, IRBuilder, Module, PointerType, Value
from tileforge.passes import AlgebraicSimplifyPass, ConstantFoldPass, CSEPass, DeadCodeEliminationPass, PassManager
from tileforge.runtime import CPUInterpreter


def generate_random_expr(builder: IRBuilder, vars_: list[Value], depth: int = 0) -> Value:
    if depth > 2 or random.random() < 0.3:
        if random.random() < 0.5:
            return random.choice(vars_)
        else:
            c_val = random.randint(1, 10)
            return builder.create_constant(float(c_val), F32)

    op_type = random.choice(["add", "sub", "mul"])
    lhs = generate_random_expr(builder, vars_, depth + 1)
    rhs = generate_random_expr(builder, vars_, depth + 1)

    if op_type == "add":
        return builder.create_add(lhs, rhs)
    elif op_type == "sub":
        return builder.create_sub(lhs, rhs)
    else:
        return builder.create_mul(lhs, rhs)


def test_fuzz_random_expression_trees():
    random.seed(42)
    np.random.seed(42)

    for test_idx in range(5):
        module = Module(f"fuzz_mod_{test_idx}")
        x = Value("%x", PointerType(F32))
        y = Value("%y", PointerType(F32))
        out = Value("%out", PointerType(F32))
        n = Value("%n", I32)

        func = Function("fuzz_kernel", [x, y, out, n], VOID)
        module.add_function(func)

        builder = IRBuilder(func.entry_block)
        pid = builder.create_program_id(0)
        c256 = builder.create_constant(256, I32)
        step1 = builder.create_mul(pid, c256)
        ar = builder.create_arange(0, 256)
        offs = builder.create_add(step1, ar)
        mask = builder.create_cmp("lt", offs, n)

        xv = builder.create_load(x, offs, mask)
        yv = builder.create_load(y, offs, mask)

        # Generate expression
        res_val = generate_random_expr(builder, [xv, yv], depth=0)
        builder.create_store(out, offs, res_val, mask)
        builder.create_return()

        # Optimize IR
        pm = PassManager([
            ConstantFoldPass(),
            AlgebraicSimplifyPass(),
            CSEPass(),
            DeadCodeEliminationPass(),
        ], verify_each=True)
        pm.run(module)

        # Execute
        N = 256
        x_np = np.random.randn(N).astype(np.float32)
        y_np = np.random.randn(N).astype(np.float32)
        out_np = np.zeros(N, dtype=np.float32)

        interpreter = CPUInterpreter()
        interpreter.execute(func, grid=(1,), args=[x_np, y_np, out_np, N])

        assert not np.isnan(out_np).any()
