import pytest
from tileforge.frontend.parser import Parser
from tileforge.frontend.semantic import SemanticAnalyzer
from tileforge.lowering.ast_to_ir import ASTToLowering
from tileforge.ir.types import PointerType, F32, I32
from tileforge.ir.printer import IRPrinter
from tileforge.ir.verifier import IRVerifier
import tileforge as tf


def test_lowering_vector_add():
    @tf.kernel
    def add(x, y, out, n):
        pid = tf.program_id(0)
        offsets = pid * 256 + tf.arange(0, 256)
        mask = offsets < n
        a = tf.load(x, offsets, mask)
        b = tf.load(y, offsets, mask)
        c = a + b
        tf.store(out, offsets, c, mask)

    parser = Parser()
    kernel_ast = parser.parse_kernel(add)
    
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32]
    analyzer = SemanticAnalyzer()
    analyzer.analyze(kernel_ast, arg_types)

    lowering = ASTToLowering()
    func = lowering.lower_kernel(kernel_ast, arg_types)

    verifier = IRVerifier()
    verifier.verify_module(lowering.module)

    printer = IRPrinter()
    ir_text = printer.print_function(func)
    assert "func @add(" in ir_text
    assert "%0 = tf.program_id axis=0 : i32" in ir_text
    assert "tf.store %out[%4], %8, mask=%5" in ir_text


def test_lowering_reassignment_creates_new_ssa_values():
    src = """
def rebind(a, b):
    x = a + b
    x = x * 2
    return x
"""
    parser = Parser()
    kernel_ast = parser.parse_kernel(src)
    arg_types = [I32, I32]
    analyzer = SemanticAnalyzer()
    analyzer.analyze(kernel_ast, arg_types)

    lowering = ASTToLowering()
    func = lowering.lower_kernel(kernel_ast, arg_types)

    verifier = IRVerifier()
    verifier.verify_module(lowering.module)

    printer = IRPrinter()
    ir_text = printer.print_function(func)
    
    # %0 = tf.add %a, %b
    # %1 = tf.constant 2
    # %2 = tf.mul %0, %1
    # tf.return %2
    assert "%0 = tf.add %a, %b : i32" in ir_text
    assert "%2 = tf.mul %0, %1 : i32" in ir_text
    assert "tf.return %2" in ir_text
