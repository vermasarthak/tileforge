"""Semantic Analysis pass: Symbol resolution, type inference, shape checking, and control flow analysis."""

from __future__ import annotations

from typing import List

from tileforge.frontend.ast_nodes import (
    Assignment,
    BinaryExpr,
    Call,
    CompareExpr,
    Expr,
    ForRangeStatement,
    IfStatement,
    KernelFunctionNode,
    Literal,
    Name,
    Return,
    Statement,
)
from tileforge.frontend.errors import (
    TypeCheckError,
    UndefinedSymbolError,
)
from tileforge.frontend.symbols import SymbolTable
from tileforge.ir.types import (
    F32,
    I1,
    I32,
    VOID,
    PointerType,
    TensorType,
    Type,
    compare_types,
    promote_types,
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
                sym = self.symbol_table.declare(
                    stmt.target, val_type, is_arg=False, line=stmt.line, column=stmt.column
                )
                if isinstance(stmt.value, Literal) and isinstance(stmt.value.value, (int, float)):
                    sym.const_value = stmt.value.value

        elif isinstance(stmt, Return):
            if stmt.value is not None:
                ret_type = self._analyze_expr(stmt.value)
                stmt.inferred_type = ret_type
            else:
                stmt.inferred_type = VOID

        elif isinstance(stmt, IfStatement):
            cond_type = self._analyze_expr(stmt.condition)
            if cond_type != I1:
                raise TypeCheckError(
                    f"If condition must evaluate to boolean scalar (i1), got {cond_type}",
                    filename=self.filename,
                    line=stmt.line,
                    column=stmt.column,
                )

            # Analyze then and else branches
            for s in stmt.then_body:
                self._analyze_statement(s)
            for s in stmt.else_body:
                self._analyze_statement(s)

        elif isinstance(stmt, ForRangeStatement):
            start_t = self._analyze_expr(stmt.start)
            end_t = self._analyze_expr(stmt.end)

            if not start_t.is_integer() or not end_t.is_integer():
                raise TypeCheckError(
                    f"For loop range bounds must be integer types, got {start_t} and {end_t}",
                    filename=self.filename,
                    line=stmt.line,
                    column=stmt.column,
                )

            # Declare loop index in current scope
            self.symbol_table.declare(stmt.var_name, I32, is_arg=False, line=stmt.line, column=stmt.column)

            for s in stmt.body:
                self._analyze_statement(s)

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
            elif isinstance(expr.value, (tuple, list, str)):
                expr.inferred_type = VOID
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
            if expr.op in ("and", "or"):
                res_t = promote_types(lhs_t, rhs_t)
            else:
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

        elif fname in {"tf.arange", "tf.range"}:
            if len(call.args) != 2:
                raise TypeCheckError(
                    "arange requires start and end arguments",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            arg0 = call.args[0]
            arg1 = call.args[1]

            start_val = None
            if isinstance(arg0, Literal) and isinstance(arg0.value, int):
                start_val = arg0.value
            elif isinstance(arg0, Name) and self.symbol_table.contains(arg0.id):
                sym = self.symbol_table.lookup(arg0.id)
                if hasattr(sym, "const_value") and isinstance(sym.const_value, int):
                    start_val = sym.const_value

            end_val = None
            if isinstance(arg1, Literal) and isinstance(arg1.value, int):
                end_val = arg1.value
            elif isinstance(arg1, Name) and self.symbol_table.contains(arg1.id):
                sym = self.symbol_table.lookup(arg1.id)
                if hasattr(sym, "const_value") and isinstance(sym.const_value, int):
                    end_val = sym.const_value

            if start_val is None or end_val is None:
                start_val = 0 if start_val is None else start_val
                end_val = 256 if end_val is None else end_val

            arg0.inferred_type = I32
            arg1.inferred_type = I32
            size = end_val - start_val
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
            offs_arg = call.args[1]

            if not isinstance(ptr_t, PointerType):
                raise TypeCheckError(
                    f"tf.load first argument must be pointer, got {ptr_t}",
                    filename=self.filename,
                    line=line,
                    column=col,
                )

            # Check offsets type (handling tuple offsets like (offs_m, k))
            if isinstance(offs_arg, Call) and offs_arg.func_name == "tuple":
                offs_elem_types = [self._analyze_expr(a) for a in offs_arg.args]
            elif isinstance(offs_arg, Literal) and isinstance(offs_arg.value, tuple):
                # Tuple literal containing Exprs or values
                offs_elem_types = [I32] * len(offs_arg.value)
            else:
                offs_t = self._analyze_expr(offs_arg)
                offs_elem_types = [offs_t]

            # Find tensor shape if any offset component is a tensor
            tensor_shape = None
            for ot in offs_elem_types:
                if isinstance(ot, TensorType):
                    tensor_shape = ot.shape
                    break

            mask_arg = call.keywords.get("mask") or (call.args[2] if len(call.args) > 2 else None)
            if mask_arg is not None:
                mask_t = self._analyze_expr(mask_arg)
                if tensor_shape is not None:
                    if not isinstance(mask_t, TensorType) or mask_t.shape != tensor_shape:
                        raise TypeCheckError(
                            f"tf.load mask shape must match offsets shape, got {mask_t} vs {tensor_shape}",
                            filename=self.filename,
                            line=line,
                            column=col,
                        )

            elem_t = ptr_t.element_type
            if tensor_shape is not None:
                return TensorType(tensor_shape, elem_t)
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
            self._analyze_expr(call.args[2])

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

        elif fname in {"tf.logical_and", "tf.logical_or"}:
            if len(call.args) != 2:
                raise TypeCheckError(
                    f"{fname} requires 2 arguments",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            a_t = self._analyze_expr(call.args[0])
            b_t = self._analyze_expr(call.args[1])
            return promote_types(a_t, b_t)

        elif fname == "tf.where":
            if len(call.args) != 3:
                raise TypeCheckError(
                    "tf.where requires condition, true_value, and false_value arguments",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            self._analyze_expr(call.args[0])
            true_t = self._analyze_expr(call.args[1])
            false_t = self._analyze_expr(call.args[2])
            return promote_types(true_t, false_t)

        elif fname in {"tf.sum", "tf.max"}:
            if len(call.args) < 1:
                raise TypeCheckError(
                    f"{fname} requires input tensor argument",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            tensor_t = self._analyze_expr(call.args[0])
            if not isinstance(tensor_t, TensorType):
                raise TypeCheckError(
                    f"{fname} argument must be TensorType, got {tensor_t}",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            return tensor_t.element_type

        elif fname == "tf.dot":
            if len(call.args) != 2:
                raise TypeCheckError(
                    "tf.dot requires 2 matrix arguments",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            a_t = self._analyze_expr(call.args[0])
            b_t = self._analyze_expr(call.args[1])
            if not isinstance(a_t, TensorType) or not isinstance(b_t, TensorType):
                raise TypeCheckError(
                    f"tf.dot arguments must be TensorType, got {a_t} and {b_t}",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            if len(a_t.shape) == 1 and len(b_t.shape) == 1:
                return TensorType((a_t.shape[0], b_t.shape[0]), a_t.element_type)
            elif len(a_t.shape) == 2 and len(b_t.shape) == 2:
                m, k1 = a_t.shape
                k2, n = b_t.shape
                if k1 != k2:
                    raise TypeCheckError(
                        f"tf.dot matrix dimension mismatch: ({m}x{k1}) @ ({k2}x{n})",
                        filename=self.filename,
                        line=line,
                        column=col,
                    )
                return TensorType((m, n), a_t.element_type)
            else:
                return TensorType((a_t.shape[0], b_t.shape[-1]), a_t.element_type)

        elif fname == "tf.zeros":
            if len(call.args) < 1:
                raise TypeCheckError("tf.zeros requires shape argument", filename=self.filename, line=line, column=col)

            # Shape tuple parsing
            shape_arg = call.args[0]
            if isinstance(shape_arg, Literal) and isinstance(shape_arg.value, (int, tuple, list)):
                shape_val = (shape_arg.value,) if isinstance(shape_arg.value, int) else tuple(shape_arg.value)
            else:
                shape_val = (256, 256)

            dtype_val = F32
            if len(call.args) > 1 and isinstance(call.args[1], Literal) and isinstance(call.args[1].value, str):
                if call.args[1].value in {"i32", "int32"}:
                    dtype_val = I32

            return TensorType(shape_val, dtype_val)

        elif fname in {"tuple", "tf.tuple"}:
            for a in call.args:
                self._analyze_expr(a)
            return VOID

        else:
            raise UndefinedSymbolError(
                f"Unrecognized TileForge builtin call '{fname}'",
                filename=self.filename,
                line=line,
                column=col,
            )
