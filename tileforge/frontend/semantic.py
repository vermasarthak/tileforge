"""Semantic Analysis pass: Symbol resolution, type inference, and shape checking."""

from __future__ import annotations
from typing import Dict, List, Optional
from tileforge.frontend.ast_nodes import (
    KernelFunctionNode,
    Argument,
    Statement,
    Assignment,
    Return,
    Expr,
    BinaryExpr,
    CompareExpr,
    Call,
    Literal,
    Name,
)
from tileforge.frontend.symbols import SymbolTable, Symbol
from tileforge.frontend.errors import (
    TypeCheckError,
    UndefinedSymbolError,
    TileForgeError,
)
from tileforge.ir.types import (
    Type,
    PrimitiveType,
    PointerType,
    TensorType,
    I1,
    I32,
    I64,
    F32,
    F64,
    VOID,
    promote_types,
    compare_types,
)


class SemanticAnalyzer:
    """Performs type inference, symbol lookup, shape verification, and builtin call validation."""
    def __init__(self, filename: str = "<kernel>"):
        self.filename: str = filename
        self.symbol_table: SymbolTable = SymbolTable()

    def analyze(self, kernel: KernelFunctionNode, arg_types: List[Type]) -> SymbolTable:
        if len(kernel.args) != len(arg_types):
            raise TypeCheckError(
                f"Kernel '{kernel.name}' expects {len(kernel.args)} arguments, got {len(arg_types)}",
                filename=self.filename,
                line=kernel.line,
                column=kernel.column,
            )

        # 1. Register function arguments in symbol table
        for arg_node, arg_type in zip(kernel.args, arg_types):
            self.symbol_table.declare(
                name=arg_node.name,
                type_=arg_type,
                is_arg=True,
                line=arg_node.line,
                column=arg_node.column,
            )
            arg_node.inferred_type = arg_type

        # 2. Analyze statements in body
        ret_type: Type = VOID
        for stmt in kernel.body:
            self._analyze_statement(stmt)
            if isinstance(stmt, Return):
                if stmt.inferred_type is not None:
                    ret_type = stmt.inferred_type

        kernel.return_type = ret_type
        return self.symbol_table

    def _analyze_statement(self, stmt: Statement) -> None:
        if isinstance(stmt, Assignment):
            val_type = self._analyze_expr(stmt.value)
            stmt.inferred_type = val_type
            if stmt.target != "_":
                # Bind or update variable in symbol table
                if self.symbol_table.contains(stmt.target):
                    # Variable re-assignment: ensure type compatibility or update binding
                    self.symbol_table.declare(
                        stmt.target, val_type, is_arg=False, line=stmt.line, column=stmt.column
                    )
                else:
                    self.symbol_table.declare(
                        stmt.target, val_type, is_arg=False, line=stmt.line, column=stmt.column
                    )

        elif isinstance(stmt, Return):
            if stmt.value is not None:
                ret_type = self._analyze_expr(stmt.value)
                stmt.inferred_type = ret_type
            else:
                stmt.inferred_type = VOID

        else:
            raise TypeCheckError(
                f"Unknown statement type '{type(stmt).__name__}'",
                filename=self.filename,
                line=stmt.line,
                column=stmt.column,
            )

    def _analyze_expr(self, expr: Expr) -> Type:
        if isinstance(expr, Literal):
            if isinstance(expr.value, bool):
                expr.inferred_type = I1
            elif isinstance(expr.value, int):
                expr.inferred_type = I32
            elif isinstance(expr.value, float):
                expr.inferred_type = F32
            else:
                raise TypeCheckError(
                    f"Unsupported literal value '{expr.value}'",
                    filename=self.filename,
                    line=expr.line,
                    column=expr.column,
                )
            return expr.inferred_type

        elif isinstance(expr, Name):
            sym = self.symbol_table.lookup(expr.id, line=expr.line, column=expr.column)
            expr.inferred_type = sym.type
            return sym.type

        elif isinstance(expr, BinaryExpr):
            lhs_t = self._analyze_expr(expr.lhs)
            rhs_t = self._analyze_expr(expr.rhs)
            res_t = promote_types(lhs_t, rhs_t)
            expr.inferred_type = res_t
            return res_t

        elif isinstance(expr, CompareExpr):
            lhs_t = self._analyze_expr(expr.lhs)
            rhs_t = self._analyze_expr(expr.rhs)
            res_t = compare_types(lhs_t, rhs_t)
            expr.inferred_type = res_t
            return res_t

        elif isinstance(expr, Call):
            res_t = self._analyze_builtin_call(expr)
            expr.inferred_type = res_t
            return res_t

        else:
            raise TypeCheckError(
                f"Unknown expression type '{type(expr).__name__}'",
                filename=self.filename,
                line=expr.line,
                column=expr.column,
            )

    def _analyze_builtin_call(self, call: Call) -> Type:
        fname = call.func_name
        line = call.line
        col = call.column

        if fname == "tf.program_id":
            if len(call.args) != 1 or not isinstance(call.args[0], Literal) or not isinstance(call.args[0].value, int):
                raise TypeCheckError(
                    "tf.program_id requires exactly 1 integer literal argument axis (e.g. tf.program_id(0))",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            axis = call.args[0].value
            if axis < 0 or axis > 2:
                raise TypeCheckError(
                    f"tf.program_id axis must be 0, 1, or 2, got {axis}",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            call.args[0].inferred_type = I32
            return I32

        elif fname == "tf.arange":
            if len(call.args) != 2:
                raise TypeCheckError(
                    "tf.arange requires start and end arguments (e.g. tf.arange(0, 256))",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            arg0 = call.args[0]
            arg1 = call.args[1]
            if not isinstance(arg0, Literal) or not isinstance(arg1, Literal):
                raise TypeCheckError(
                    "tf.arange arguments must be static integer literals",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            start = arg0.value
            end = arg1.value
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                raise TypeCheckError(
                    f"tf.arange range invalid: [{start}, {end})",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            arg0.inferred_type = I32
            arg1.inferred_type = I32
            size = end - start
            return TensorType((size,), I32)

        elif fname == "tf.load":
            if len(call.args) < 2:
                raise TypeCheckError(
                    "tf.load requires pointer and offsets arguments",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            ptr_t = self._analyze_expr(call.args[0])
            offs_t = self._analyze_expr(call.args[1])

            if not isinstance(ptr_t, PointerType):
                raise TypeCheckError(
                    f"tf.load first argument must be pointer, got {ptr_t}",
                    filename=self.filename,
                    line=line,
                    column=col,
                )

            mask_arg = call.keywords.get("mask") or (call.args[2] if len(call.args) > 2 else None)
            if mask_arg is not None:
                mask_t = self._analyze_expr(mask_arg)
                if isinstance(offs_t, TensorType):
                    if not isinstance(mask_t, TensorType) or mask_t.shape != offs_t.shape:
                        raise TypeCheckError(
                            f"tf.load mask shape must match offsets shape, got {mask_t} vs {offs_t}",
                            filename=self.filename,
                            line=line,
                            column=col,
                        )

            elem_t = ptr_t.element_type
            if isinstance(offs_t, TensorType):
                return TensorType(offs_t.shape, elem_t)
            return elem_t

        elif fname == "tf.store":
            if len(call.args) < 3:
                raise TypeCheckError(
                    "tf.store requires pointer, offsets, and value arguments",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            ptr_t = self._analyze_expr(call.args[0])
            offs_t = self._analyze_expr(call.args[1])
            val_t = self._analyze_expr(call.args[2])

            if not isinstance(ptr_t, PointerType):
                raise TypeCheckError(
                    f"tf.store first argument must be pointer, got {ptr_t}",
                    filename=self.filename,
                    line=line,
                    column=col,
                )

            mask_arg = call.keywords.get("mask") or (call.args[3] if len(call.args) > 3 else None)
            if mask_arg is not None:
                self._analyze_expr(mask_arg)

            return VOID

        elif fname == "tf.where":
            if len(call.args) != 3:
                raise TypeCheckError(
                    "tf.where requires condition, true_value, and false_value arguments",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            cond_t = self._analyze_expr(call.args[0])
            true_t = self._analyze_expr(call.args[1])
            false_t = self._analyze_expr(call.args[2])
            return promote_types(true_t, false_t)

        else:
            raise UndefinedSymbolError(
                f"Unrecognized TileForge builtin call '{fname}'",
                filename=self.filename,
                line=line,
                column=col,
            )
