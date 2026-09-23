"""Dominance Analysis and Control Flow Graph (CFG) analysis for TileForge IR functions."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from tileforge.ir.block import Block
from tileforge.ir.function import Function
from tileforge.ir.operation import Operation
from tileforge.ir.value import Value


class DominanceInfo:
    """Computes reachability, RPO, dominators, immediate dominators, and SSA dominance queries."""
    def __init__(self, func: Function):
        self.func: Function = func
        self.reachable_blocks: List[Block] = []
        self.preds: Dict[Block, List[Block]] = {}
        self.succs: Dict[Block, List[Block]] = {}
        self.rpo: List[Block] = []
        self.dom: Dict[Block, Set[Block]] = {}
        self.idom: Dict[Block, Optional[Block]] = {}
        self.df: Dict[Block, Set[Block]] = {}

        self._compute()

    def _compute(self) -> None:
        if not self.func.blocks:
            return

        entry = self.func.entry_block

        # 1. Compute reachable blocks via BFS
        visited: Set[Block] = set()
        queue: List[Block] = [entry]
        visited.add(entry)

        while queue:
            curr = queue.pop(0)
            self.reachable_blocks.append(curr)
            for succ in curr.successors:
                if succ.parent_function == self.func and succ not in visited:
                    visited.add(succ)
                    queue.append(succ)

        # 2. Build predecessors and successors for reachable blocks
        for b in self.reachable_blocks:
            self.preds[b] = []
            self.succs[b] = [s for s in b.successors if s in visited]

        for b in self.reachable_blocks:
            for s in self.succs[b]:
                if b not in self.preds[s]:
                    self.preds[s].append(b)

        # 3. Compute Reverse Postorder (RPO)
        rpo_visited: Set[Block] = set()
        post_order: List[Block] = []

        def dfs(b: Block) -> None:
            rpo_visited.add(b)
            for s in self.succs[b]:
                if s not in rpo_visited:
                    dfs(s)
            post_order.append(b)

        dfs(entry)
        self.rpo = list(reversed(post_order))

        # 4. Compute Dominator Sets using iterative fixed-point algorithm
        all_reachable = set(self.reachable_blocks)
        for b in self.reachable_blocks:
            if b == entry:
                self.dom[b] = {entry}
            else:
                self.dom[b] = set(all_reachable)

        changed = True
        while changed:
            changed = False
            for b in self.rpo:
                if b == entry:
                    continue
                pred_list = [p for p in self.preds[b] if p in self.dom]
                if not pred_list:
                    new_dom = {b}
                else:
                    new_dom = set.intersection(*(self.dom[p] for p in pred_list))
                    new_dom.add(b)

                if new_dom != self.dom[b]:
                    self.dom[b] = new_dom
                    changed = True

        # 5. Compute Immediate Dominators (idom)
        for b in self.reachable_blocks:
            if b == entry:
                self.idom[b] = None
                continue
            strict_doms = self.dom[b] - {b}
            # idom is the strict dominator d that is dominated by all other strict dominators
            idom_candidate = None
            for d in strict_doms:
                # Check if d is dominated by all other strict dominators of b
                if all(other in self.dom[d] for other in strict_doms):
                    idom_candidate = d
                    break
            self.idom[b] = idom_candidate

        # 6. Compute Dominance Frontiers (df)
        for b in self.reachable_blocks:
            self.df[b] = set()

        for b in self.reachable_blocks:
            if len(self.preds[b]) >= 2:
                for p in self.preds[b]:
                    runner: Optional[Block] = p
                    while runner is not None and runner != self.idom[b]:
                        self.df[runner].add(b)
                        runner = self.idom[runner]

    def dominates(self, a: Block, b: Block) -> bool:
        """Returns True if block `a` dominates block `b`."""
        if b not in self.dom:
            return False
        return a in self.dom[b]

    def dominates_value_use(self, val: Value, use_op: Operation) -> bool:
        """Returns True if SSA `val` definition dominates `use_op`."""
        use_block = use_op.parent_block
        if use_block is None:
            return False

        # Function argument: always dominates all uses in function
        if val in self.func.args:
            return True

        # Block argument: dominates if argument's block dominates use_block
        def_block = None
        for b in self.func.blocks:
            if val in b.args:
                def_block = b
                break

        if def_block is not None:
            if def_block == use_block:
                return True
            return self.dominates(def_block, use_block)

        # Value defined by an operation
        def_op = val.defining_op
        if def_op is None or def_op.parent_block is None:
            return False

        def_block = def_op.parent_block
        if def_block == use_block:
            # Inside same block: defining op index must precede use op index
            try:
                def_idx = def_block.operations.index(def_op)
                use_idx = def_block.operations.index(use_op)
                return def_idx < use_idx
            except ValueError:
                return False

        # Cross-block: defining block must strictly dominate use block
        return self.dominates(def_block, use_block)
