"""Function representation for TileForge SSA IR."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from tileforge.ir.block import Block

if TYPE_CHECKING:
    from tileforge.ir.module import Module
    from tileforge.ir.types import Type
    from tileforge.ir.value import Value


class Function:
    """Represents a kernel function in TileForge IR."""
    def __init__(
        self,
        name: str,
        args: List[Value],
        return_type: Type,
        parent_module: Optional[Module] = None,
    ):
        self.name: str = name
        self.args: List[Value] = args
        self.return_type: Type = return_type
        self.blocks: List[Block] = []
        self.parent_module: Optional[Module] = parent_module

        # Create default entry block
        entry_block = Block(name="entry", parent_function=self)
        self.blocks.append(entry_block)

    @property
    def entry_block(self) -> Block:
        return self.blocks[0]

    def create_block(self, name: Optional[str] = None, args: Optional[List[Value]] = None) -> Block:
        if name is None:
            name = f"bb{len(self.blocks)}"
        block = Block(name=name, parent_function=self, args=args)
        self.blocks.append(block)
        return block

    def __repr__(self) -> str:
        args_str = ", ".join(f"{a.name}: {a.type}" for a in self.args)
        return f"Function({self.name}({args_str}) -> {self.return_type})"
