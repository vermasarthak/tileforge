"""Custom AST node hierarchy for TileForge language representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tileforge.ir.types import Type


@dataclass
class ASTNode:
    """Base class for all TileForge AST nodes."""
    line: int = 0
    column: int = 0
    inferred_type: Optional[Type] = field(default=None, repr=False)


@dataclass
class Expr(ASTNode):
    """Base class for expression AST nodes."""
    pass


@dataclass
class Statement(ASTNode):
    """Base class for statement AST nodes."""
    pass


@dataclass
class Literal(Expr):
    value: Any = None


@dataclass
class Name(Expr):
    id: str = ""


@dataclass
class Argument(ASTNode):
    name: str = ""
    type_annotation: Optional[str] = None


@dataclass
class BinaryExpr(Expr):
    op: str = ""  # '+', '-', '*', '/', '//'
    lhs: Expr = field(default_factory=Expr)
    rhs: Expr = field(default_factory=Expr)


@dataclass
class CompareExpr(Expr):
    op: str = ""  # '<', '<=', '>', '>=', '==', '!='
    lhs: Expr = field(default_factory=Expr)
    rhs: Expr = field(default_factory=Expr)


@dataclass
class Call(Expr):
    func_name: str = ""
    args: List[Expr] = field(default_factory=list)
    keywords: Dict[str, Expr] = field(default_factory=dict)


@dataclass
class Assignment(Statement):
    target: str = ""
    value: Expr = field(default_factory=Expr)


@dataclass
class Return(Statement):
    value: Optional[Expr] = None


@dataclass
class IfStatement(Statement):
    condition: Expr = field(default_factory=Expr)
    then_body: List[Statement] = field(default_factory=list)
    else_body: List[Statement] = field(default_factory=list)


@dataclass
class ForRangeStatement(Statement):
    var_name: str = ""
    start: Expr = field(default_factory=Expr)
    end: Expr = field(default_factory=Expr)
    body: List[Statement] = field(default_factory=list)


@dataclass
class KernelFunctionNode(ASTNode):
    name: str = ""
    args: List[Argument] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    return_type: Optional[Type] = field(default=None, repr=False)

