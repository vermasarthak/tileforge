"""Dead Code Elimination (DCE) pass for TileForge IR."""

from __future__ import annotations
from tileforge.ir.module import Module
from tileforge.ir.operation import Operation
from tileforge.passes.manager import Pass


class DeadCodeEliminationPass(Pass):
    """Removes unused pure operations without side-effects."""
    def run(self, module: Module) -> bool:
        modified_any = False
        changed = True

        # Run to fixed point
        while changed:
            changed = False
            for func in module.functions:
                for block in func.blocks:
                    i = 0
                    while i < len(block.operations):
                        op = block.operations[i]
                        if op.is_pure() and op.results:
                            # Check if all results are unused
                            if all(not res.is_used() for res in op.results):
                                op.erase()
                                changed = True
                                modified_any = True
                                continue
                        i += 1
        return modified_any
