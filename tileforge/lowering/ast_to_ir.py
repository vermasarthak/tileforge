"""SSA Lowering pass: Converts TileForge AST nodes into typed SSA IR."""

from __future__ import annotations
from typing import Dict, List, Optional
from tileforge.frontend.ast_nodes import (
    KernelFunctionNode,
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
from tileforge.frontend.errors import LoweringError
from tileforge.ir.types import Type, PointerType, TensorType, VOID, I32
from tileforge.ir.value import Value
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.block import Block
from tileforge.ir.function import Function
from tileforge.ir.module import Module
from tileforge.ir.builder import IRBuilder


class ASTToLowering:
    """Translates semantically analyzed TileForge AST into SSA IR."""
    def __init__(self, module_name: str = "main"):
        self.module: Module = Module(module_name)

    def lower_kernel(self, kernel: KernelFunctionNode, arg_types: List[Type]) -> Function:
        # 1. Create function argument SSA values
        fn_args: List[Value] = []
        symbol_env: Dict[str, Value] = {}

        for arg_node, arg_type in zip(kernel.args, arg_types):
            arg_val = Value(name=f"%{arg_node.name}", type_=arg_type)
            fn_args.append(arg_val)
            symbol_env[arg_node.name] = arg_val

        # 2. Construct IR function and builder
        ret_type = kernel.return_type or VOID
        func = Function(name=kernel.name, args=fn_args, return_type=ret_type, parent_module=self.module)
        self.module.add_function(func)
        
        builder = IRBuilder(block=func.entry_block)

        # 3. Lower AST statements sequentially
        has_return = False
        for stmt in kernel.body:
            if isinstance(stmt, Assignment):
                val_res = self._lower_expr(stmt.value, builder, symbol_env)
                if stmt.target != "_":
                    # Update symbol binding to NEW SSA Value (SSA versioning)
                    symbol_env[stmt.target] = val_res

            elif isinstance(stmt, Return):
                ret_val = self._lower_expr(stmt.value, builder, symbol_env) if stmt.value is not None else None
                builder.create_return(ret_val)
                has_return = True

        if not has_return:
            builder.create_return()

        return func

    def _lower_expr(self, expr: Expr, builder: IRBuilder, symbol_env: Dict[str, Value]) -> Value:
        if isinstance(expr, Literal):
            val_t = expr.inferred_type
            if val_t is None:
                raise LoweringError(f"AST node {expr} missing inferred_type", line=expr.line, column=expr.column)
            return builder.create_constant(expr.value, val_t)

        elif isinstance(expr, Name):
            if expr.id not in symbol_env:
                raise LoweringError(f"Undefined SSA binding for variable '{expr.id}'", line=expr.line, column=expr.column)
            return symbol_env[expr.id]

        elif isinstance(expr, BinaryExpr):
            lhs_val = self._lower_expr(expr.lhs, builder, symbol_env)
            rhs_val = self._lower_expr(expr.rhs, builder, symbol_env)
            
            if expr.op in ("+", "or"):
                return builder.create_add(lhs_val, rhs_val)
            elif expr.op == "-":
                return builder.create_sub(lhs_val, rhs_val)
            elif expr.op in ("*", "and"):
                return builder.create_mul(lhs_val, rhs_val)
            elif expr.op in ("/", "//"):
                return builder.create_div(lhs_val, rhs_val)
            else:
                raise LoweringError(f"Unsupported binary operator '{expr.op}'", line=expr.line, column=expr.column)

        elif isinstance(expr, CompareExpr):
            lhs_val = self._lower_expr(expr.lhs, builder, symbol_env)
            rhs_val = self._lower_expr(expr.rhs, builder, symbol_env)
            
            pred_map = {
                "<": "lt",
                "<=": "le",
                ">": "gt",
                ">=": "ge",
                "==": "eq",
                "!=": "ne",
            }
            if expr.op not in pred_map:
                raise LoweringError(f"Unsupported comparison operator '{expr.op}'", line=expr.line, column=expr.column)
            return builder.create_cmp(pred_map[expr.op], lhs_val, rhs_val)

        elif isinstance(expr, Call):
            return self._lower_builtin_call(expr, builder, symbol_env)

        else:
            raise LoweringError(f"Cannot lower unknown expression '{type(expr).__name__}'", line=expr.line, column=expr.column)

    def _lower_builtin_call(self, call: Call, builder: IRBuilder, symbol_env: Dict[str, Value]) -> Optional[Value]:
        fname = call.func_name
        line = call.line
        col = call.column

        if fname == "tf.program_id":
            axis = call.args[0].value
            return builder.create_program_id(axis)

        elif fname == "tf.arange":
            start = call.args[0].value
            end = call.args[1].value
            return builder.create_arange(start, end)

        elif fname == "tf.load":
            ptr_val = self._lower_expr(call.args[0], builder, symbol_env)
            offs_val = self._lower_expr(call.args[1], builder, symbol_env)
            
            mask_arg = call.keywords.get("mask") or (call.args[2] if len(call.args) > 2 else None)
            mask_val = self._lower_expr(mask_arg, builder, symbol_env) if mask_arg is not None else None
            
            return builder.create_load(ptr_val, offs_val, mask_val)

        elif fname == "tf.store":
            ptr_val = self._lower_expr(call.args[0], builder, symbol_env)
            offs_val = self._lower_expr(call.args[1], builder, symbol_env)
            val_to_store = self._lower_expr(call.args[2], builder, symbol_env)
            
            mask_arg = call.keywords.get("mask") or (call.args[3] if len(call.args) > 3 else None)
            mask_val = self._lower_expr(mask_arg, builder, symbol_env) if mask_arg is not None else None
            
            builder.create_store(ptr_val, offs_val, val_to_store, mask_val)
            return None

        elif fname == "tf.where":
            cond_val = self._lower_expr(call.args[0], builder, symbol_env)
            t_val = self._lower_expr(call.args[1], builder, symbol_env)
            f_val = self._lower_expr(call.args[2], builder, symbol_env)
            return builder.create_where(cond_val, t_val, f_val)

        else:
            raise LoweringError(f"Unrecognized builtin call '{fname}'", line=line, column=col)
