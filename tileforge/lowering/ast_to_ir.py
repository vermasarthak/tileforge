"""SSA Lowering pass: Converts TileForge AST nodes into typed SSA IR with multi-block CFG control flow."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

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
from tileforge.frontend.errors import LoweringError
from tileforge.ir.block import Block
from tileforge.ir.builder import IRBuilder
from tileforge.ir.function import Function
from tileforge.ir.module import Module
from tileforge.ir.operation import OpType
from tileforge.ir.types import F32, I32, VOID, TensorType, Type, promote_types
from tileforge.ir.value import Value


class ASTToLowering:
    """Translates semantically analyzed TileForge AST into multi-block SSA IR with block arguments."""
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

        builder = IRBuilder(func.entry_block)

        # 3. Lower AST statements sequentially
        self._lower_statement_list(kernel.body, func, builder, symbol_env)

        # Ensure active block has terminator
        if builder.block and (not builder.block.operations or not builder.block.operations[-1].is_terminator()):
            builder.create_return()

        return func

    def _lower_statement_list(
        self,
        stmts: List[Statement],
        func: Function,
        builder: IRBuilder,
        symbol_env: Dict[str, Value],
    ) -> None:
        for stmt in stmts:
            if builder.block and builder.block.operations and builder.block.operations[-1].is_terminator():
                # Unreachable statements after terminator
                break

            if isinstance(stmt, Assignment):
                val_res = self._lower_expr(stmt.value, builder, symbol_env)
                if stmt.target != "_":
                    symbol_env[stmt.target] = val_res

            elif isinstance(stmt, Return):
                ret_val = self._lower_expr(stmt.value, builder, symbol_env) if stmt.value is not None else None
                builder.create_return(ret_val)

            elif isinstance(stmt, IfStatement):
                self._lower_if_statement(stmt, func, builder, symbol_env)

            elif isinstance(stmt, ForRangeStatement):
                self._lower_for_range_statement(stmt, func, builder, symbol_env)

            else:
                raise LoweringError(f"Unknown statement type '{type(stmt).__name__}'", line=stmt.line, column=stmt.column)

    def _lower_if_statement(
        self,
        stmt: IfStatement,
        func: Function,
        builder: IRBuilder,
        symbol_env: Dict[str, Value],
    ) -> None:
        cond_val = self._lower_expr(stmt.condition, builder, symbol_env)
        curr_block = builder.block

        then_block = func.create_block(name=f"then_{len(func.blocks)}")
        else_block = func.create_block(name=f"else_{len(func.blocks)}")
        merge_block = func.create_block(name=f"merge_{len(func.blocks)}")

        # Evaluate then-branch in then_block
        then_env = dict(symbol_env)
        builder.set_insertion_point(then_block)
        self._lower_statement_list(stmt.then_body, func, builder, then_env)
        then_final_block = builder.block

        # Evaluate else-branch in else_block
        else_env = dict(symbol_env)
        builder.set_insertion_point(else_block)
        self._lower_statement_list(stmt.else_body, func, builder, else_env)
        else_final_block = builder.block

        # Identify modified variables across branches that need block arguments in merge_block
        merge_var_names = sorted(set(then_env.keys()) | set(else_env.keys()))
        differing_vars: List[str] = []
        for var in merge_var_names:
            v_then = then_env.get(var)
            v_else = else_env.get(var)
            if v_then != v_else and v_then is not None and v_else is not None:
                differing_vars.append(var)

        # Create merge block arguments
        then_branch_args: List[Value] = []
        else_branch_args: List[Value] = []
        for var in differing_vars:
            v_t = then_env[var]
            v_e = else_env[var]
            common_type = promote_types(v_t.type, v_e.type)

            if v_t.type != common_type and isinstance(common_type, TensorType):
                builder.set_insertion_point(then_final_block)
                v_t_bc = builder.create_constant(0.0 if common_type.element_type.is_float() else 0, common_type)
                then_branch_args.append(v_t_bc)
            else:
                then_branch_args.append(v_t)

            if v_e.type != common_type and isinstance(common_type, TensorType):
                builder.set_insertion_point(else_final_block)
                v_e_bc = builder.create_constant(0.0 if common_type.element_type.is_float() else 0, common_type)
                else_branch_args.append(v_e_bc)
            else:
                else_branch_args.append(v_e)

            b_arg = builder.create_block_arg(merge_block, common_type, name_prefix=var)
            symbol_env[var] = b_arg

        # Terminate curr_block with cond_br
        builder.set_insertion_point(curr_block)
        builder.create_cond_br(cond_val, then_block, else_block)

        # Terminate then/else final blocks with branch to merge_block
        if then_final_block and (not then_final_block.operations or not then_final_block.operations[-1].is_terminator()):
            builder.set_insertion_point(then_final_block)
            builder.create_br(merge_block, then_branch_args)

        if else_final_block and (not else_final_block.operations or not else_final_block.operations[-1].is_terminator()):
            builder.set_insertion_point(else_final_block)
            builder.create_br(merge_block, else_branch_args)

        builder.set_insertion_point(merge_block)

    def _lower_for_range_statement(
        self,
        stmt: ForRangeStatement,
        func: Function,
        builder: IRBuilder,
        symbol_env: Dict[str, Value],
    ) -> None:
        start_val = self._lower_expr(stmt.start, builder, symbol_env)
        end_val = self._lower_expr(stmt.end, builder, symbol_env)
        curr_block = builder.block

        # Collect target variable names assigned in loop body
        modified_in_body: Set[str] = set()
        def _collect_targets(stmts: List[Statement]):
            for s in stmts:
                if isinstance(s, Assignment):
                    modified_in_body.add(s.target)
                elif isinstance(s, IfStatement):
                    _collect_targets(s.then_body)
                    _collect_targets(s.else_body)
                elif isinstance(s, ForRangeStatement):
                    _collect_targets(s.body)
        _collect_targets(stmt.body)

        loop_carried_vars = sorted([v for v in symbol_env.keys() if v in modified_in_body])

        header_block = func.create_block(name=f"loop_header_{len(func.blocks)}")
        body_block = func.create_block(name=f"loop_body_{len(func.blocks)}")
        exit_block = func.create_block(name=f"loop_exit_{len(func.blocks)}")

        # Pre-pass: simulate loop body lowering to infer promoted loop-carried variable types
        dummy_body_env = dict(symbol_env)
        dummy_body_env[stmt.var_name] = Value("dummy_idx", I32)
        for var in loop_carried_vars:
            dummy_body_env[var] = Value(f"dummy_{var}", symbol_env[var].type)

        saved_block_count = len(func.blocks)
        isolated_builder = IRBuilder()
        isolated_builder.block = Block(name="dummy_body")
        self._lower_statement_list(stmt.body, func, isolated_builder, dummy_body_env)

        # Completely remove any dummy blocks added to func during pre-pass
        while len(func.blocks) > saved_block_count:
            b = func.blocks.pop()
            b.operations.clear()

        # Compute promoted types
        promoted_carried_types: List[Type] = []
        init_carried_vals: List[Value] = []
        for var in loop_carried_vars:
            init_val = symbol_env[var]
            init_carried_vals.append(init_val)
            body_val = dummy_body_env[var]
            promoted_carried_types.append(promote_types(init_val.type, body_val.type))

        # Create header block arguments using promoted types
        index_arg = builder.create_block_arg(header_block, I32, name_prefix=stmt.var_name)
        header_carried_args: List[Value] = []
        promoted_init_vals: List[Value] = []

        for idx, var in enumerate(loop_carried_vars):
            init_val = init_carried_vals[idx]
            target_type = promoted_carried_types[idx]
            h_arg = builder.create_block_arg(header_block, target_type, name_prefix=var)
            header_carried_args.append(h_arg)

            if init_val.type != target_type and isinstance(target_type, TensorType):
                builder.set_insertion_point(curr_block)
                promoted_const = builder.create_constant(0.0 if target_type.element_type.is_float() else 0, target_type)
                promoted_init_vals.append(promoted_const)
            else:
                promoted_init_vals.append(init_val)

        # Branch from curr_block to header_block
        builder.set_insertion_point(curr_block)
        builder.create_br(header_block, [start_val] + promoted_init_vals)

        # Build header block condition check (%i < end)
        builder.set_insertion_point(header_block)
        cond_val = builder.create_cmp("lt", index_arg, end_val)
        builder.create_cond_br(cond_val, body_block, exit_block, then_args=[], else_args=header_carried_args)

        # Build body block execution
        body_env = dict(symbol_env)
        body_env[stmt.var_name] = index_arg
        for var, h_arg in zip(loop_carried_vars, header_carried_args):
            body_env[var] = h_arg

        builder.set_insertion_point(body_block)
        self._lower_statement_list(stmt.body, func, builder, body_env)
        body_final_block = builder.block

        # Increment index (%i + 1)
        builder.set_insertion_point(body_final_block)
        c1 = builder.create_constant(1, I32)
        next_index = builder.create_add(index_arg, c1)

        next_carried_vals: List[Value] = []
        for idx, var in enumerate(loop_carried_vars):
            b_val = body_env[var]
            h_type = header_carried_args[idx].type
            if b_val.type != h_type and isinstance(h_type, TensorType):
                # Broadcast/expand scalar b_val to tensor
                expanded_val = builder.create_constant(0.0 if h_type.element_type.is_float() else 0, h_type)
                next_carried_vals.append(expanded_val)
            else:
                next_carried_vals.append(b_val)

        builder.create_br(header_block, [next_index] + next_carried_vals)

        # Build exit block
        exit_args: List[Value] = []
        for var, h_arg in zip(loop_carried_vars, header_carried_args):
            e_arg = builder.create_block_arg(exit_block, h_arg.type, name_prefix=var)
            exit_args.append(e_arg)
            symbol_env[var] = e_arg

        builder.set_insertion_point(exit_block)

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

            if expr.op == "+":
                return builder.create_add(lhs_val, rhs_val)
            elif expr.op == "-":
                return builder.create_sub(lhs_val, rhs_val)
            elif expr.op == "*":
                return builder.create_mul(lhs_val, rhs_val)
            elif expr.op in ("/", "//"):
                return builder.create_div(lhs_val, rhs_val)
            elif expr.op == "and":
                return builder.create_logical_and(lhs_val, rhs_val)
            elif expr.op == "or":
                return builder.create_logical_or(lhs_val, rhs_val)
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

        elif fname in {"tf.arange", "tf.range"}:
            start_arg = call.args[0]
            end_arg = call.args[1]

            start_val = start_arg.value if isinstance(start_arg, Literal) and isinstance(start_arg.value, int) else 0
            end_val = end_arg.value if isinstance(end_arg, Literal) and isinstance(end_arg.value, int) else 256

            if isinstance(end_arg, Name) and end_arg.id in symbol_env:
                v = symbol_env[end_arg.id]
                if v.defining_op and v.defining_op.op_type == OpType.CONSTANT:
                    end_val = int(v.defining_op.attributes.get("value", 256))

            if isinstance(start_arg, Name) and start_arg.id in symbol_env:
                v = symbol_env[start_arg.id]
                if v.defining_op and v.defining_op.op_type == OpType.CONSTANT:
                    start_val = int(v.defining_op.attributes.get("value", 0))

            return builder.create_arange(start_val, end_val)

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

        elif fname == "tuple":
            return tuple(self._lower_expr(a, builder, symbol_env) for a in call.args)

        elif fname in {"tf.logical_and", "tf.logical_or"}:
            a_val = self._lower_expr(call.args[0], builder, symbol_env)
            b_val = self._lower_expr(call.args[1], builder, symbol_env)
            if fname == "tf.logical_and":
                return builder.create_logical_and(a_val, b_val)
            else:
                return builder.create_logical_or(a_val, b_val)

        elif fname == "tf.where":
            cond_val = self._lower_expr(call.args[0], builder, symbol_env)
            t_val = self._lower_expr(call.args[1], builder, symbol_env)
            f_val = self._lower_expr(call.args[2], builder, symbol_env)
            return builder.create_where(cond_val, t_val, f_val)

        elif fname == "tf.sum":
            tensor_val = self._lower_expr(call.args[0], builder, symbol_env)
            return builder.create_reduce_sum(tensor_val)

        elif fname == "tf.max":
            tensor_val = self._lower_expr(call.args[0], builder, symbol_env)
            return builder.create_reduce_max(tensor_val)

        elif fname == "tf.dot":
            a_val = self._lower_expr(call.args[0], builder, symbol_env)
            b_val = self._lower_expr(call.args[1], builder, symbol_env)
            return builder.create_dot(a_val, b_val)

        elif fname == "tf.zeros":
            shape_arg = call.args[0]
            if isinstance(shape_arg, Literal) and isinstance(shape_arg.value, (int, tuple, list)):
                shape_tuple = (shape_arg.value,) if isinstance(shape_arg.value, int) else tuple(shape_arg.value)
            else:
                shape_tuple = (256, 256)

            dtype_t = F32
            if len(call.args) > 1 and isinstance(call.args[1], Literal) and isinstance(call.args[1].value, str):
                if call.args[1].value in {"i32", "int32"}:
                    dtype_t = I32

            res_type = TensorType(shape_tuple, dtype_t)
            return builder.create_constant(0.0 if dtype_t == F32 else 0, res_type)

        else:
            raise LoweringError(f"Unrecognized builtin call '{fname}'", line=line, column=col)
