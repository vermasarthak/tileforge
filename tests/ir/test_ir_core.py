import pytest

from tileforge.frontend.errors import IRVerificationError
from tileforge.ir import (
    F32,
    I32,
    VOID,
    Function,
    IRBuilder,
    IRPrinter,
    IRVerifier,
    Module,
    PointerType,
    Value,
)


def test_ir_building_printing_verifying():
    module = Module("test_module")
    x = Value("%x", PointerType(F32))
    y = Value("%y", PointerType(F32))
    out = Value("%out", PointerType(F32))
    n = Value("%n", I32)

    func = Function("add", [x, y, out, n], VOID)
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
    zv = builder.create_add(xv, yv)
    builder.create_store(out, offs, zv, mask)
    builder.create_return()

    verifier = IRVerifier()
    verifier.verify_module(module)

    printer = IRPrinter()
    text = printer.print_module(module)
    assert "func @add(" in text
    assert "%0 = tf.program_id axis=0 : i32" in text
    assert "tf.store %out[%4], %8, mask=%5" in text


def test_verifier_rejects_invalid_ir():
    module = Module("bad_module")
    x = Value("%x", I32)
    func = Function("bad", [x], VOID)
    module.add_function(func)

    builder = IRBuilder(func.entry_block)
    # Undefined value use
    undef = Value("%undef", I32)
    builder.create_add(x, undef)
    builder.create_return()

    verifier = IRVerifier()
    with pytest.raises(IRVerificationError):
        verifier.verify_module(module)
