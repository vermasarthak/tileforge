import pytest
from tileforge.frontend.parser import Parser
from tileforge.frontend.errors import UnsupportedSyntaxError, TileForgeSyntaxError
from tileforge.frontend.ast_nodes import KernelFunctionNode, Assignment, BinaryExpr, Call
import tileforge as tf


def test_parse_valid_kernel():
    @tf.kernel
    def add(x, y, out, n):
        pid = tf.program_id(0)
        offsets = pid * 256 + tf.arange(0, 256)
        mask = offsets < n
        a = tf.load(x, offsets, mask)
        b = tf.load(y, offsets, mask)
        tf.store(out, offsets, a + b, mask)

    parser = Parser()
    ast_node = parser.parse_kernel(add)
    assert ast_node.name == "add"
    assert len(ast_node.args) == 4
    assert len(ast_node.body) == 6


def test_parse_rejects_loops():
    src = """
def kernel(x):
    for i in range(10):
        pass
"""
    parser = Parser()
    with pytest.raises(UnsupportedSyntaxError) as exc_info:
        parser.parse_kernel(src)
    assert "Unsupported Python statement 'For'" in str(exc_info.value)


def test_parse_rejects_arbitrary_python():
    src = """
def kernel(x):
    import os
    os.system("echo bad")
"""
    parser = Parser()
    with pytest.raises(UnsupportedSyntaxError):
        parser.parse_kernel(src)
