import pytest
from tileforge.ir import (
    Module, Function, Value, IRBuilder, PointerType, F32, I32, I1, VOID, IRVerifier
)
from tileforge.frontend.errors import IRVerificationError


def test_verifier_rejects_undominated_use():
    module = Module("test_undominated")
    cond = Value("%cond", I1)
    func = Function("fn", [cond], VOID)
    module.add_function(func)

    b_then = func.create_block("then")
    b_else = func.create_block("else")
    b_merge = func.create_block("merge")

    # Entry branches to then/else
    builder = IRBuilder(func.entry_block)
    builder.create_cond_br(cond, b_then, b_else)

    # In then branch, define %0
    builder.set_insertion_point(b_then)
    c10 = builder.create_constant(10, I32)
    builder.create_br(b_merge)

    # In else branch, branch to merge
    builder.set_insertion_point(b_else)
    builder.create_br(b_merge)

    # In merge, try to use %0 without block arguments!
    builder.set_insertion_point(b_merge)
    builder.create_add(c10, c10)
    builder.create_return()

    verifier = IRVerifier()
    with pytest.raises(IRVerificationError) as exc_info:
        verifier.verify_module(module)
    assert "not dominated by its definition" in str(exc_info.value)


def test_verifier_rejects_branch_arg_count_mismatch():
    module = Module("test_arg_mismatch")
    func = Function("fn", [], VOID)
    module.add_function(func)

    b_target = func.create_block("target")
    builder = IRBuilder(b_target)
    arg0 = builder.create_block_arg(b_target, I32)
    builder.create_return()

    # Entry branches with 0 args, but target expects 1 arg!
    builder.set_insertion_point(func.entry_block)
    builder.create_br(b_target, dest_args=[])

    verifier = IRVerifier()
    with pytest.raises(IRVerificationError) as exc_info:
        verifier.verify_module(module)
    assert "Branch argument count" in str(exc_info.value)


def test_verifier_rejects_branch_arg_type_mismatch():
    module = Module("test_type_mismatch")
    func = Function("fn", [], VOID)
    module.add_function(func)

    b_target = func.create_block("target")
    builder = IRBuilder(b_target)
    arg0 = builder.create_block_arg(b_target, I32)
    builder.create_return()

    builder.set_insertion_point(func.entry_block)
    c_float = builder.create_constant(1.0, F32)
    builder.create_br(b_target, dest_args=[c_float])

    verifier = IRVerifier()
    with pytest.raises(IRVerificationError) as exc_info:
        verifier.verify_module(module)
    assert "type" in str(exc_info.value).lower()
