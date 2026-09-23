"""CPU Reference Interpreter for TileForge IR using NumPy arrays with multi-block CFG support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tileforge.frontend.errors import InterpreterError
from tileforge.ir.block import Block
from tileforge.ir.function import Function
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.types import TensorType
from tileforge.ir.value import Value


class CPUInterpreter:
    """Executes TileForge SSA IR on CPU across grid blocks using NumPy memory buffers."""

    def execute(
        self,
        func: Function,
        grid: Tuple[int, ...],
        args: List[Any],
    ) -> None:
        if len(args) != len(func.args):
            raise InterpreterError(
                f"Kernel execution expected {len(func.args)} arguments, got {len(args)}"
            )

        grid_x = grid[0] if len(grid) >= 1 else 1
        grid_y = grid[1] if len(grid) >= 2 else 1
        grid_z = grid[2] if len(grid) >= 3 else 1

        # Dispatch kernel once per grid program ID
        for gz in range(grid_z):
            for gy in range(grid_y):
                for gx in range(grid_x):
                    self._execute_single_launch(func, (gx, gy, gz), args)

    def _execute_single_launch(
        self,
        func: Function,
        grid_pos: Tuple[int, int, int],
        args: List[Any],
    ) -> None:
        # SSA value environment: maps Value -> python scalar or numpy array
        env: Dict[Value, Any] = {}

        # Bind function arguments
        for arg_val, arg_input in zip(func.args, args):
            env[arg_val] = arg_input

        # Multi-block CFG execution
        curr_block: Optional[Block] = func.entry_block
        visited_count: int = 0
        max_ops: int = 1_000_000

        while curr_block is not None:
            next_block: Optional[Block] = None

            for op in curr_block.operations:
                visited_count += 1
                if visited_count > max_ops:
                    raise InterpreterError("Infinite loop or exceeded maximum execution op threshold")

                if op.op_type == OpType.BR:
                    target_block = op.successors[0]
                    for b_arg, val_opnd in zip(target_block.args, op.operands):
                        env[b_arg] = env[val_opnd]
                    next_block = target_block
                    break

                elif op.op_type == OpType.COND_BR:
                    cond_val = env[op.operands[0]]
                    then_block = op.successors[0]
                    else_block = op.successors[1]

                    t_count = op.attributes.get("then_arg_count", 0)
                    e_count = op.attributes.get("else_arg_count", 0)
                    t_opnds = op.operands[1:1 + t_count]
                    e_opnds = op.operands[1 + t_count:1 + t_count + e_count]

                    if bool(cond_val):
                        for b_arg, val_opnd in zip(then_block.args, t_opnds):
                            env[b_arg] = env[val_opnd]
                        next_block = then_block
                    else:
                        for b_arg, val_opnd in zip(else_block.args, e_opnds):
                            env[b_arg] = env[val_opnd]
                        next_block = else_block
                    break

                elif op.op_type == OpType.RETURN:
                    return

                else:
                    self._execute_operation(op, grid_pos, env)

            curr_block = next_block

    def _execute_operation(
        self,
        op: Operation,
        grid_pos: Tuple[int, int, int],
        env: Dict[Value, Any],
    ) -> None:
        if op.op_type == OpType.CONSTANT:
            val = op.attributes["value"]
            res_type = op.results[0].type
            if isinstance(res_type, TensorType):
                fill_val = float(val) if res_type.element_type.is_float() else int(val)
                env[op.results[0]] = np.full(res_type.shape, fill_val, dtype=np.float32 if res_type.element_type.is_float() else np.int32)
            else:
                env[op.results[0]] = val

        elif op.op_type == OpType.PROGRAM_ID:
            axis = op.attributes.get("axis", 0)
            if axis < 0 or axis > 2:
                raise InterpreterError(f"Invalid program_id axis {axis}")
            env[op.results[0]] = grid_pos[axis]

        elif op.op_type == OpType.ARANGE:
            start = op.attributes["start"]
            end = op.attributes["end"]
            env[op.results[0]] = np.arange(start, end, dtype=np.int32)

        elif op.op_type in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}:
            v0 = env[op.operands[0]]
            v1 = env[op.operands[1]]
            if op.op_type == OpType.ADD:
                res = v0 + v1
            elif op.op_type == OpType.SUB:
                res = v0 - v1
            elif op.op_type == OpType.MUL:
                res = v0 * v1
            elif op.op_type == OpType.DIV:
                res = v0 // v1 if isinstance(v0, (int, np.integer)) and isinstance(v1, (int, np.integer)) else v0 / v1
            env[op.results[0]] = res

        elif op.op_type == OpType.LOGICAL_AND:
            v0 = env[op.operands[0]]
            v1 = env[op.operands[1]]
            env[op.results[0]] = np.logical_and(v0, v1) if isinstance(v0, np.ndarray) or isinstance(v1, np.ndarray) else (v0 and v1)

        elif op.op_type == OpType.LOGICAL_OR:
            v0 = env[op.operands[0]]
            v1 = env[op.operands[1]]
            env[op.results[0]] = np.logical_or(v0, v1) if isinstance(v0, np.ndarray) or isinstance(v1, np.ndarray) else (v0 or v1)

        elif op.op_type == OpType.CMP:
            v0 = env[op.operands[0]]
            v1 = env[op.operands[1]]
            pred = op.attributes["predicate"]
            if pred == "lt":
                res = v0 < v1
            elif pred == "le":
                res = v0 <= v1
            elif pred == "gt":
                res = v0 > v1
            elif pred == "ge":
                res = v0 >= v1
            elif pred == "eq":
                res = v0 == v1
            elif pred == "ne":
                res = v0 != v1
            else:
                raise InterpreterError(f"Unknown comparison predicate '{pred}'")
            env[op.results[0]] = res

        elif op.op_type == OpType.LOAD:
            ptr = env[op.operands[0]]
            if not isinstance(ptr, np.ndarray):
                raise InterpreterError(f"Load source buffer must be numpy ndarray, got {type(ptr)}")

            has_mask = op.attributes.get("has_mask", False)
            if has_mask:
                mask = env[op.operands[-1]]
                offset_operands = op.operands[1:-1]
            else:
                mask = None
                offset_operands = op.operands[1:]

            if len(offset_operands) == 1 and isinstance(env[offset_operands[0]], tuple):
                offs_vals = tuple(env[x] if isinstance(x, Value) else x for x in env[offset_operands[0]])
            else:
                offs_vals = tuple(env[op_val] for op_val in offset_operands)

            if len(offs_vals) == 2:
                o0, o1 = offs_vals[0], offs_vals[1]
                if isinstance(o0, np.ndarray) and isinstance(o1, np.ndarray):
                    m0 = np.logical_and(o0 >= 0, o0 < ptr.shape[0])
                    m1 = np.logical_and(o1 >= 0, o1 < ptr.shape[1])
                    grid_m0, grid_m1 = np.meshgrid(m0, m1, indexing="ij")
                    valid_2d = np.logical_and(grid_m0, grid_m1)
                    if mask is not None:
                        valid_2d = np.logical_and(valid_2d, mask)

                    res = np.zeros((len(o0), len(o1)), dtype=ptr.dtype)
                    idx0, idx1 = np.meshgrid(o0, o1, indexing="ij")
                    res[valid_2d] = ptr[idx0[valid_2d], idx1[valid_2d]]
                elif isinstance(o0, np.ndarray):
                    valid = np.logical_and(o0 >= 0, o0 < ptr.shape[0])
                    res = np.zeros(len(o0), dtype=ptr.dtype)
                    res[valid] = ptr[o0[valid], o1]
                elif isinstance(o1, np.ndarray):
                    valid = np.logical_and(o1 >= 0, o1 < ptr.shape[1])
                    res = np.zeros(len(o1), dtype=ptr.dtype)
                    res[valid] = ptr[o0, o1[valid]]
                else:
                    res = ptr[o0, o1]
            else:
                offs = offs_vals[0]
                if mask is not None:
                    if isinstance(offs, np.ndarray):
                        # Masked load: unmasked elements are set to 0.0 (or neutral float)
                        res = np.zeros(offs.shape, dtype=ptr.dtype)
                        valid_mask = np.logical_and(mask, offs >= 0)
                        valid_mask = np.logical_and(valid_mask, offs < len(ptr))
                        res[valid_mask] = ptr[offs[valid_mask]]
                    else:
                        res = ptr[offs] if mask else 0
                else:
                    res = ptr[offs]

            env[op.results[0]] = res

        elif op.op_type == OpType.STORE:
            ptr = env[op.operands[0]]
            if not isinstance(ptr, np.ndarray):
                raise InterpreterError(f"Store target buffer must be numpy ndarray, got {type(ptr)}")

            has_mask = op.attributes.get("has_mask", False)
            if has_mask:
                mask = env[op.operands[-1]]
                val = env[op.operands[-2]]
                offset_operands = op.operands[1:-2]
            else:
                mask = None
                val = env[op.operands[-1]]
                offset_operands = op.operands[1:-1]

            if len(offset_operands) == 1 and isinstance(env[offset_operands[0]], tuple):
                offs_vals = tuple(env[x] if isinstance(x, Value) else x for x in env[offset_operands[0]])
            else:
                offs_vals = tuple(env[op_val] for op_val in offset_operands)

            if len(offs_vals) == 2:
                o0, o1 = offs_vals[0], offs_vals[1]
                if isinstance(o0, np.ndarray) and isinstance(o1, np.ndarray):
                    m0 = np.logical_and(o0 >= 0, o0 < ptr.shape[0])
                    m1 = np.logical_and(o1 >= 0, o1 < ptr.shape[1])
                    grid_m0, grid_m1 = np.meshgrid(m0, m1, indexing="ij")
                    valid_2d = np.logical_and(grid_m0, grid_m1)
                    if mask is not None:
                        valid_2d = np.logical_and(valid_2d, mask)

                    idx0, idx1 = np.meshgrid(o0, o1, indexing="ij")
                    val_arr = np.broadcast_to(val, idx0.shape)
                    ptr[idx0[valid_2d], idx1[valid_2d]] = val_arr[valid_2d]
                else:
                    ptr[o0, o1] = val
            else:
                offs = offs_vals[0]
                if mask is not None:
                    if isinstance(offs, np.ndarray):
                        valid_mask = np.logical_and(mask, offs >= 0)
                        valid_mask = np.logical_and(valid_mask, offs < len(ptr))
                        val_arr = np.broadcast_to(val, offs.shape) if not isinstance(val, np.ndarray) else val
                        ptr[offs[valid_mask]] = val_arr[valid_mask]
                    else:
                        if mask and 0 <= offs < len(ptr):
                            ptr[offs] = val
                else:
                    ptr[offs] = val

        elif op.op_type == OpType.WHERE:
            cond = env[op.operands[0]]
            v_t = env[op.operands[1]]
            v_f = env[op.operands[2]]
            env[op.results[0]] = np.where(cond, v_t, v_f)

        elif op.op_type == OpType.REDUCE_SUM:
            t_val = env[op.operands[0]]
            env[op.results[0]] = np.sum(t_val)

        elif op.op_type == OpType.REDUCE_MAX:
            t_val = env[op.operands[0]]
            env[op.results[0]] = np.max(t_val)

        elif op.op_type == OpType.DOT:
            a_val = env[op.operands[0]]
            b_val = env[op.operands[1]]
            if isinstance(a_val, np.ndarray) and isinstance(b_val, np.ndarray) and a_val.ndim == 1 and b_val.ndim == 1:
                env[op.results[0]] = np.outer(a_val, b_val)
            else:
                env[op.results[0]] = np.matmul(a_val, b_val)

        else:
            raise InterpreterError(f"Unsupported interpreter op_type '{op.op_type}'")
