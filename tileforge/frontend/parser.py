"""Python AST to TileForge AST Parser with strict syntax validation."""

from __future__ import annotations
import ast
import inspect
from typing import Callable, Any, List, Union

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
from tileforge.frontend.errors import TileForgeSyntaxError, UnsupportedSyntaxError
from tileforge.language.decorators import KernelFunction


class Parser:
    """Parses Python source code into validated TileForge AST nodes."""
    def __init__(self, filename: str = "<kernel>"):
        self.filename: str = filename

    def parse_kernel(self, kernel_obj: Union[KernelFunction, Callable[..., Any], str]) -> KernelFunctionNode:
        source_code = ""
        if isinstance(kernel_obj, KernelFunction):
            source_code = kernel_obj._source_code
        elif callable(kernel_obj):
            try:
                source_code = inspect.getsource(kernel_obj)
            except Exception as e:
                raise TileForgeSyntaxError(f"Could not extract source code from kernel function: {e}")
        elif isinstance(kernel_obj, str):
            source_code = kernel_obj
        else:
            raise TileForgeSyntaxError(f"Invalid kernel object type: {type(kernel_obj)}")

        # Deduce indentation if source was extracted from a method or nested function
        source_code = inspect.cleandoc(source_code)

        try:
            py_ast = ast.parse(source_code, filename=self.filename)
        except SyntaxError as e:
            raise TileForgeSyntaxError(
                f"Python syntax error: {e.msg}",
                filename=self.filename,
                line=e.lineno,
                column=e.offset,
            )

        if not isinstance(py_ast, ast.Module) or not py_ast.body:
            raise TileForgeSyntaxError("Empty or invalid Python AST module", filename=self.filename)

        # Look for function definition (ignore decorator on top)
        func_def: ast.FunctionDef | None = None
        for stmt in py_ast.body:
            if isinstance(stmt, ast.FunctionDef):
                func_def = stmt
                break

        if func_def is None:
            raise TileForgeSyntaxError("No function definition found in kernel source", filename=self.filename)

        return self._parse_function_def(func_def)

    def _parse_function_def(self, node: ast.FunctionDef) -> KernelFunctionNode:
        fn_name = node.name
        args: List[Argument] = []

        for arg in node.args.args:
            args.append(Argument(
                name=arg.arg,
                line=arg.lineno,
                column=arg.col_offset,
            ))

        body_stmts: List[Statement] = []
        for stmt in node.body:
            body_stmts.append(self._parse_statement(stmt))

        return KernelFunctionNode(
            name=fn_name,
            args=args,
            body=body_stmts,
            line=node.lineno,
            column=node.col_offset,
        )

    def _parse_statement(self, stmt: ast.stmt) -> Statement:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise UnsupportedSyntaxError(
                    "Only single-variable assignment (e.g., x = expr) is supported",
                    filename=self.filename,
                    line=stmt.lineno,
                    column=stmt.col_offset,
                )
            target_name = stmt.targets[0].id
            val_expr = self._parse_expr(stmt.value)
            return Assignment(
                target=target_name,
                value=val_expr,
                line=stmt.lineno,
                column=stmt.col_offset,
            )

        elif isinstance(stmt, ast.Expr):
            # Side-effecting expressions like tf.store(...)
            expr = self._parse_expr(stmt.value)
            if not isinstance(expr, Call):
                raise UnsupportedSyntaxError(
                    "Standalone expressions must be builtin function calls (e.g. tf.store)",
                    filename=self.filename,
                    line=stmt.lineno,
                    column=stmt.col_offset,
                )
            return Assignment(
                target="_",
                value=expr,
                line=stmt.lineno,
                column=stmt.col_offset,
            )

        elif isinstance(stmt, ast.Return):
            val_expr = self._parse_expr(stmt.value) if stmt.value is not None else None
            return Return(
                value=val_expr,
                line=stmt.lineno,
                column=stmt.col_offset,
            )

        else:
            raise UnsupportedSyntaxError(
                f"Unsupported Python statement '{type(stmt).__name__}'. "
                "Only assignments, returns, and builtin calls are allowed.",
                filename=self.filename,
                line=stmt.lineno,
                column=stmt.col_offset,
            )

    def _parse_expr(self, expr: ast.expr) -> Expr:
        line = getattr(expr, "lineno", 0)
        col = getattr(expr, "col_offset", 0)

        if isinstance(expr, ast.Constant):
            if not isinstance(expr.value, (int, float, bool)):
                raise UnsupportedSyntaxError(
                    f"Unsupported literal type: {type(expr.value).__name__}",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            return Literal(value=expr.value, line=line, column=col)

        elif isinstance(expr, ast.Name):
            return Name(id=expr.id, line=line, column=col)

        elif isinstance(expr, ast.BinOp):
            op_map = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.FloorDiv: "//",
            }
            op_type = type(expr.op)
            if op_type not in op_map:
                raise UnsupportedSyntaxError(
                    f"Unsupported binary operator '{op_type.__name__}'",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            lhs = self._parse_expr(expr.left)
            rhs = self._parse_expr(expr.right)
            return BinaryExpr(op=op_map[op_type], lhs=lhs, rhs=rhs, line=line, column=col)

        elif isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                raise UnsupportedSyntaxError(
                    "Chained comparisons are not supported",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            cmp_op_map = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            op_type = type(expr.ops[0])
            if op_type not in cmp_op_map:
                raise UnsupportedSyntaxError(
                    f"Unsupported comparison operator '{op_type.__name__}'",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            lhs = self._parse_expr(expr.left)
            rhs = self._parse_expr(expr.comparators[0])
            return CompareExpr(op=cmp_op_map[op_type], lhs=lhs, rhs=rhs, line=line, column=col)

        elif isinstance(expr, ast.Call):
            # Resolve function name: tf.program_id or program_id
            func_name = ""
            if isinstance(expr.func, ast.Attribute):
                if isinstance(expr.func.value, ast.Name) and expr.func.value.id in {"tf", "tileforge"}:
                    func_name = f"tf.{expr.func.attr}"
                else:
                    func_name = f"{self._expr_to_str(expr.func.value)}.{expr.func.attr}"
            elif isinstance(expr.func, ast.Name):
                func_name = f"tf.{expr.func.id}" if not expr.func.id.startswith("tf.") else expr.func.id
            else:
                raise UnsupportedSyntaxError(
                    "Unsupported function call target",
                    filename=self.filename,
                    line=line,
                    column=col,
                )

            pos_args = [self._parse_expr(a) for a in expr.args]
            kwargs = {kw.arg: self._parse_expr(kw.value) for kw in expr.keywords if kw.arg is not None}

            return Call(
                func_name=func_name,
                args=pos_args,
                keywords=kwargs,
                line=line,
                column=col,
            )

        elif isinstance(expr, ast.BoolOp):
            bool_op_map = {
                ast.And: "and",
                ast.Or: "or",
            }
            op_type = type(expr.op)
            if op_type not in bool_op_map:
                raise UnsupportedSyntaxError(
                    f"Unsupported boolean operator '{op_type.__name__}'",
                    filename=self.filename,
                    line=line,
                    column=col,
                )
            # Fold chained bool ops left to right
            res_expr = self._parse_expr(expr.values[0])
            for val in expr.values[1:]:
                rhs = self._parse_expr(val)
                res_expr = BinaryExpr(op=bool_op_map[op_type], lhs=res_expr, rhs=rhs, line=line, column=col)
            return res_expr

        else:
            raise UnsupportedSyntaxError(
                f"Unsupported Python expression '{type(expr).__name__}'",
                filename=self.filename,
                line=line,
                column=col,
            )

    def _expr_to_str(self, expr: ast.expr) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        return "<expr>"
