from tileforge.ir import I32, VOID, Function, IRBuilder, IRPrinter, Module, Value
from tileforge.passes import AlgebraicSimplifyPass, ConstantFoldPass, CSEPass, DeadCodeEliminationPass, PassManager


def test_constant_folding():
    module = Module("test_fold")
    func = Function("fn", [], I32)
    module.add_function(func)

    builder = IRBuilder(func.entry_block)
    c3 = builder.create_constant(3, I32)
    c4 = builder.create_constant(4, I32)
    add_val = builder.create_add(c3, c4)
    builder.create_return(add_val)

    pm = PassManager([ConstantFoldPass(), DeadCodeEliminationPass()])
    pm.run(module)

    printer = IRPrinter()
    ir_text = printer.print_module(module)
    assert "tf.constant 7" in ir_text
    assert "tf.add" not in ir_text


def test_algebraic_simplification():
    module = Module("test_algebraic")
    x = Value("%x", I32)
    func = Function("fn", [x], I32)
    module.add_function(func)

    builder = IRBuilder(func.entry_block)
    c0 = builder.create_constant(0, I32)
    c1 = builder.create_constant(1, I32)
    step1 = builder.create_add(x, c0)  # x + 0 -> x
    step2 = builder.create_mul(step1, c1)  # x * 1 -> x
    builder.create_return(step2)

    pm = PassManager([AlgebraicSimplifyPass(), DeadCodeEliminationPass()])
    pm.run(module)

    printer = IRPrinter()
    ir_text = printer.print_module(module)
    assert "tf.return %x" in ir_text
    assert "tf.add" not in ir_text
    assert "tf.mul" not in ir_text


def test_cse_pass():
    module = Module("test_cse")
    a = Value("%a", I32)
    b = Value("%b", I32)
    func = Function("fn", [a, b], I32)
    module.add_function(func)

    builder = IRBuilder(func.entry_block)
    val1 = builder.create_add(a, b)
    val2 = builder.create_add(a, b)
    final_val = builder.create_mul(val1, val2)
    builder.create_return(final_val)

    pm = PassManager([CSEPass()])
    pm.run(module)

    printer = IRPrinter()
    ir_text = printer.print_module(module)
    assert ir_text.count("tf.add") == 1
    assert "%0 = tf.add %a, %b" in ir_text or "tf.add %a, %b" in ir_text


def test_dce_pass():
    module = Module("test_dce")
    a = Value("%a", I32)
    b = Value("%b", I32)
    func = Function("fn", [a, b], VOID)
    module.add_function(func)

    builder = IRBuilder(func.entry_block)
    unused = builder.create_add(a, b)
    builder.create_return()

    pm = PassManager([DeadCodeEliminationPass()])
    pm.run(module)

    printer = IRPrinter()
    ir_text = printer.print_module(module)
    assert "tf.add" not in ir_text
