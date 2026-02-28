"""TileForge optimization passes package."""

from tileforge.passes.manager import Pass, PassManager
from tileforge.passes.constant_fold import ConstantFoldPass
from tileforge.passes.algebraic import AlgebraicSimplifyPass
from tileforge.passes.dce import DeadCodeEliminationPass
from tileforge.passes.cse import CSEPass

__all__ = [
    "Pass",
    "PassManager",
    "ConstantFoldPass",
    "AlgebraicSimplifyPass",
    "DeadCodeEliminationPass",
    "CSEPass",
]
