import pytest
from tileforge.ir import Module, Function, Value, IRBuilder, IRPrinter, I32, I1, VOID
from tileforge.passes import PassManager, SimplifyCFGPass, ConstantFoldPass


def test_simplify_cfg_constant_branch():
    module = Module("test_simplify")
    func = Function("fn", [], VOID)
    module.add_function(func)

    b_then = func.create_block("then")
    b_else = func.create_block("else")

    builder = IRBuilder(func.entry_block)
    c_true = builder.create_constant(True, I1)
    builder.create_cond_br(c_true, b_then, b_else)

    builder.set_insertion_point(b_then)
    builder.create_return()

    builder.set_insertion_point(b_else)
    builder.create_return()

    pm = PassManager([SimplifyCFGPass()])
    pm.run(module)

    printer = IRPrinter()
    ir_text = printer.print_module(module)
    assert "^else" not in ir_text
    assert "tf.br ^then" in ir_text
