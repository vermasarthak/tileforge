import pytest
from tileforge.ir import Module, Function, Value, IRBuilder, I32, I1, VOID
from tileforge.ir.dominance import DominanceInfo


def test_dominance_diamond_cfg():
    module = Module("test_dom")
    cond = Value("%cond", I1)
    func = Function("fn", [cond], VOID)
    module.add_function(func)

    b_then = func.create_block("then")
    b_else = func.create_block("else")
    b_merge = func.create_block("merge")

    builder = IRBuilder(func.entry_block)
    builder.create_cond_br(cond, b_then, b_else)

    builder.set_insertion_point(b_then)
    builder.create_br(b_merge)

    builder.set_insertion_point(b_else)
    builder.create_br(b_merge)

    builder.set_insertion_point(b_merge)
    builder.create_return()

    dom_info = DominanceInfo(func)

    entry = func.entry_block
    assert dom_info.dominates(entry, entry)
    assert dom_info.dominates(entry, b_then)
    assert dom_info.dominates(entry, b_else)
    assert dom_info.dominates(entry, b_merge)

    assert not dom_info.dominates(b_then, b_merge)
    assert not dom_info.dominates(b_else, b_merge)
    assert dom_info.idom[b_merge] == entry
