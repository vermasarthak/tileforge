import pytest
from tileforge.frontend.parser import Parser
from tileforge.frontend.semantic import SemanticAnalyzer
from tileforge.frontend.errors import TypeCheckError, UndefinedSymbolError
from tileforge.ir.types import PointerType, F32, I32, TensorType
import tileforge as tf


def test_semantic_analysis_vector_add():
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
    analyzer = SemanticAnalyzer()
    
    arg_types = [PointerType(F32), PointerType(F32), PointerType(F32), I32]
    sym_table = analyzer.analyze(kernel_ast, arg_types)
    
    c_sym = sym_table.lookup("c")
    assert isinstance(c_sym.type, TensorType)
    assert c_sym.type.element_type == F32
    assert c_sym.type.shape == (256,)


def test_semantic_rejects_undefined_symbol():
    src = """
def bad(x):
    y = x + undefined_var
"""
    parser = Parser()
    ast_node = parser.parse_kernel(src)
    analyzer = SemanticAnalyzer()
    with pytest.raises(UndefinedSymbolError):
        analyzer.analyze(ast_node, [I32])


def test_semantic_rejects_invalid_builtin_args():
    src = """
def bad(x):
    pid = tf.program_id(99)
"""
    parser = Parser()
    ast_node = parser.parse_kernel(src)
    analyzer = SemanticAnalyzer()
    with pytest.raises(TypeCheckError):
        analyzer.analyze(ast_node, [I32])
