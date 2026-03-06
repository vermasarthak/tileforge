# Dominance & Control Flow Analysis in TileForge

TileForge incorporates custom dominance analysis (`DominanceInfo`) to analyze function control-flow graphs and enforce SSA scoping invariants.

## Dominance Definitions

1. **Dominators (`dom[B]`)**: A block `A` dominates a block `B` (`A dom B`) if every path from the function entry block to `B` must pass through `A`.
2. **Immediate Dominator (`idom[B]`)**: The unique strict dominator of `B` that does not strictly dominate any other strict dominator of `B`.
3. **Dominance Frontier (`df[B]`)**: The set of all blocks `Y` such that `B` dominates a predecessor of `Y`, but `B` does not strictly dominate `Y`.

## Dominance Algorithm

TileForge computes dominator sets iteratively using the classic fixed-point intersection algorithm over Reverse Postorder (RPO) block orderings:

```text
dom[entry] = {entry}
for each block B != entry:
    dom[B] = set of all reachable blocks

changed = True
while changed:
    changed = False
    for each block B in RPO (excluding entry):
        new_dom = {B} U (Intersection of dom[P] for all predecessors P of B)
        if new_dom != dom[B]:
            dom[B] = new_dom
            changed = True
```

## SSA Verifier Enforcement

`IRVerifier` queries `DominanceInfo.dominates_value_use(val, use_op)` to ensure:
- An SSA value `%x` defined in block `A` can ONLY be used in block `B` if `A` dominates `B`.
- Within the same block `A`, the defining instruction of `%x` must precede the consuming instruction.
- Values defined inside sibling `then`/`else` blocks cannot leak into a merge block without passing through block arguments.
