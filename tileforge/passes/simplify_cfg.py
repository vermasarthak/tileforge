"""SimplifyCFGPass: Folds constant branches and eliminates unreachable basic blocks."""

from __future__ import annotations
from typing import Set, List, Optional
from tileforge.ir.module import Module
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.block import Block
from tileforge.passes.manager import Pass


class SimplifyCFGPass(Pass):
    """Eliminates dead control-flow branches and unreachable basic blocks."""
    def run(self, module: Module) -> bool:
        modified = False
        for func in module.functions:
            # 1. Constant branch folding
            for block in func.blocks:
                term = block.terminator
                if term is not None and term.op_type == OpType.COND_BR:
                    cond_val = term.operands[0]
                    if cond_val.defining_op is not None and cond_val.defining_op.op_type == OpType.CONSTANT:
                        c_val = cond_val.defining_op.attributes.get("value")
                        if isinstance(c_val, (bool, int)):
                            then_block = term.successors[0]
                            else_block = term.successors[1]
                            t_count = term.attributes.get("then_arg_count", 0)
                            e_count = term.attributes.get("else_arg_count", 0)
                            
                            t_args = term.operands[1:1 + t_count]
                            e_args = term.operands[1 + t_count:1 + t_count + e_count]

                            # Replace cond_br with br
                            term.erase()
                            if c_val:
                                target_block = then_block
                                branch_args = t_args
                            else:
                                target_block = else_block
                                branch_args = e_args

                            br_op = Operation(
                                OpType.BR,
                                operands=branch_args,
                                results=[],
                                parent_block=block,
                                successors=[target_block],
                            )
                            block.operations.append(br_op)
                            modified = True

            # 2. Unreachable block elimination
            if len(func.blocks) > 1:
                entry = func.entry_block
                reachable: Set[Block] = set()
                queue: List[Block] = [entry]
                reachable.add(entry)

                while queue:
                    curr = queue.pop(0)
                    for succ in curr.successors:
                        if succ.parent_function == func and succ not in reachable:
                            reachable.add(succ)
                            queue.append(succ)

                unreachable_blocks = [b for b in func.blocks if b not in reachable]
                if unreachable_blocks:
                    for ub in unreachable_blocks:
                        # Erase ops in unreachable block
                        for op in list(ub.operations):
                            op.erase()
                        func.blocks.remove(ub)
                    modified = True

        return modified
