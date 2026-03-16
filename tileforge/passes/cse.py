"""Common Subexpression Elimination (CSE) pass for TileForge IR."""

from __future__ import annotations
from typing import Dict, Tuple, Any
from tileforge.ir.module import Module
from tileforge.ir.operation import Operation, OpType
from tileforge.passes.manager import Pass


class CSEPass(Pass):
    """Eliminates duplicate pure computations within basic blocks."""
    def run(self, module: Module) -> bool:
        modified = False
        for func in module.functions:
            for block in func.blocks:
                seen_exprs: Dict[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, Any], ...]], Operation] = {}
                i = 0
                while i < len(block.operations):
                    op = block.operations[i]
                    if op.is_pure() and op.results:
                        key = self._make_op_key(op)
                        if key in seen_exprs:
                            existing_op = seen_exprs[key]
                            # Replace uses of op.results[0] with existing_op.results[0]
                            op.results[0].replace_all_uses_with(existing_op.results[0])
                            op.erase()
                            modified = True
                            continue
                        else:
                            seen_exprs[key] = op
                    i += 1
        return modified

    def _make_op_key(self, op: Operation) -> Tuple[str, Tuple[str, ...], Tuple[Tuple[str, Any], ...], Tuple[str, ...]]:
        opnd_names = tuple(o.name for o in op.operands)
        res_types = tuple(str(r.type) for r in op.results)
        # Sort attributes for canonical dict equality key
        attr_pairs = tuple(sorted((k, repr(v)) for k, v in op.attributes.items()))
        return (op.op_type, opnd_names, attr_pairs, res_types)
