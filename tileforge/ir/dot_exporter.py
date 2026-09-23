"""Graphviz .dot export utility for TileForge CFG visualization."""

from __future__ import annotations

from tileforge.ir.function import Function
from tileforge.ir.printer import IRPrinter


def export_cfg_dot(func: Function) -> str:
    """Exports Graphviz DOT representation of function CFG."""
    printer = IRPrinter()
    lines = [f"digraph G_{func.name} {{", '  node [shape=box, fontname="Courier"];']

    for block in func.blocks:
        block_body = printer.print_block(block).replace('"', '\\"')
        label = block_body.replace("\n", "\\l") + "\\l"
        lines.append(f'  "{block.name}" [label="{label}"];')

    for block in func.blocks:
        for succ in block.successors:
            lines.append(f'  "{block.name}" -> "{succ.name}";')

    lines.append("}")
    return "\n".join(lines)
