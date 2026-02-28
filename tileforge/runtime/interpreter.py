"""CPU Reference Interpreter for TileForge IR using NumPy arrays."""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from tileforge.ir.module import Module
from tileforge.ir.function import Function
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.value import Value
from tileforge.frontend.errors import InterpreterError


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

        # Execute blocks starting with entry block
        for block in func.blocks:
            for op in block.operations:
                self._execute_operation(op, grid_pos, env)

    def _execute_operation(
        self,
        op: Operation,
        grid_pos: Tuple[int, int, int],
        env: Dict[Value, Any],
    ) -> None:
        if op.op_type == OpType.CONSTANT:
            val = op.attributes["value"]
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
            offs = env[op.operands[1]]
            mask = env[op.operands[2]] if len(op.operands) > 2 else None

            if not isinstance(ptr, np.ndarray):
                raise InterpreterError(f"Load source buffer must be numpy ndarray, got {type(ptr)}")

            if mask is not None:
                # Masked load: for mask=False positions, fill with default zero
                if isinstance(offs, np.ndarray):
                    res = np.zeros(offs.shape, dtype=ptr.dtype)
                    # Filter valid index positions where mask is True
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
            offs = env[op.operands[1]]
            val = env[op.operands[2]]
            mask = env[op.operands[3]] if len(op.operands) > 3 else None

            if not isinstance(ptr, np.ndarray):
                raise InterpreterError(f"Store target buffer must be numpy ndarray, got {type(ptr)}")

            if mask is not None:
                # Masked store: write ONLY at indices where mask is True and within buffer range
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

        elif op.op_type == OpType.RETURN:
            pass

        else:
            raise InterpreterError(f"Unsupported interpreter op_type '{op.op_type}'")
