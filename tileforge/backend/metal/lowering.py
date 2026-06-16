"""Lowering pass: High-Level TileForge SSA IR -> GPU Backend IR."""

from __future__ import annotations
from typing import Dict, List, Optional
from tileforge.ir.module import Module as HLModule
from tileforge.ir.function import Function as HLFunction
from tileforge.ir.block import Block as HLBlock
from tileforge.ir.operation import Operation as HLOperation, OpType as HLOpType
from tileforge.ir.value import Value as HLValue
from tileforge.ir.types import Type, PointerType, TensorType, PrimitiveType, I32, I1, F32, VOID

from tileforge.backend.metal.ir import (
    GPUModule,
    GPUFunction,
    GPUBlock,
    GPUOperation,
    GPUOpType,
    GPUValue,
    GPUKernelArg,
    AddressSpace,
)


class HLToGPULowering:
    """Lowers high-level tensor SSA IR into GPU backend lane/threadgroup IR."""

    def __init__(self):
        self.val_map: Dict[HLValue, GPUValue] = {}
        self.block_map: Dict[HLBlock, GPUBlock] = {}
        self.var_count = 0

    def new_var_name(self, prefix: str = "g") -> str:
        self.var_count += 1
        return f"{prefix}{self.var_count}"

    def lower_module(self, hl_module: HLModule) -> GPUModule:
        gpu_mod = GPUModule()
        for func in hl_module.functions:
            gpu_func = self.lower_function(func)
            gpu_mod.add_function(gpu_func)
        return gpu_mod

    def lower_function(self, hl_func: HLFunction) -> GPUFunction:
        gpu_args: List[GPUKernelArg] = []
        for arg in hl_func.args:
            addr_space = AddressSpace.GLOBAL if isinstance(arg.type, PointerType) else AddressSpace.CONSTANT
            gpu_args.append(GPUKernelArg(arg.name, arg.type, addr_space))

        gpu_func = GPUFunction(hl_func.name, gpu_args)

        dot_op = None
        for block in hl_func.blocks:
            for op in block.operations:
                if op.op_type == HLOpType.DOT:
                    dot_op = op
                    break
            if dot_op: break

        if dot_op:
            bm = dot_op.attributes.get("BM", 16)
            bn = dot_op.attributes.get("BN", 16)
            gpu_func.grid_dimensions = (2,)
            gpu_func.threadgroup_dimensions = (bn, bm, 1)
        else:
            gpu_func.grid_dimensions = (1,)
            gpu_func.threadgroup_dimensions = (256, 1, 1)

        # First pass: create all GPU blocks and map block arguments
        for hl_b in hl_func.blocks:
            gpu_b = gpu_func.create_block(hl_b.name)
            self.block_map[hl_b] = gpu_b

        # Populate block arguments and function parameters in entry block
        for hl_arg, gpu_arg in zip(hl_func.args, gpu_func.args):
            g_val = GPUValue(gpu_arg.name, gpu_arg.type, gpu_arg.address_space)
            self.val_map[hl_arg] = g_val

        for hl_b in hl_func.blocks:
            gpu_b = self.block_map[hl_b]
            for arg in hl_b.args:
                elem_type = arg.type.element_type if isinstance(arg.type, TensorType) else arg.type
                g_arg = GPUValue(self.new_var_name(arg.name), elem_type)
                gpu_b.args.append(g_arg)
                self.val_map[arg] = g_arg

        # Lower operations in each block
        for hl_b in hl_func.blocks:
            gpu_b = self.block_map[hl_b]
            for op in hl_b.operations:
                self.lower_operation(op, gpu_b, gpu_func, hl_func)

        return gpu_func

    def lower_operation(self, op: HLOperation, gpu_block: GPUBlock, gpu_func: GPUFunction, hl_func: HLFunction) -> None:
        if op.op_type == HLOpType.PROGRAM_ID:
            axis = op.attributes.get("axis", 0)
            res_val = GPUValue(self.new_var_name("pid"), I32)
            self.val_map[op.results[0]] = res_val
            gpu_block.append_operation(
                GPUOperation(
                    GPUOpType.PROGRAM_ID,
                    operands=[],
                    results=[res_val],
                    attributes={"axis": axis},
                )
            )

        elif op.op_type == HLOpType.CONSTANT:
            hl_res = op.results[0]
            val = op.attributes["value"]
            scalar_type = hl_res.type.element_type if isinstance(hl_res.type, TensorType) else hl_res.type
            res_val = GPUValue(self.new_var_name("c"), scalar_type)
            self.val_map[hl_res] = res_val
            gpu_block.append_operation(
                GPUOperation(
                    GPUOpType.CONSTANT,
                    operands=[],
                    results=[res_val],
                    attributes={"value": val},
                )
            )

        elif op.op_type == HLOpType.ARANGE:
            hl_res = op.results[0]
            start = op.attributes.get("start", 0)
            end = op.attributes.get("end", 256)
            
            tid_val = GPUValue(self.new_var_name("tid"), I32)
            gpu_block.append_operation(
                GPUOperation(GPUOpType.THREAD_ID, operands=[], results=[tid_val], attributes={"axis": 0})
            )
            
            if start != 0:
                c_start = GPUValue(self.new_var_name("c_start"), I32)
                gpu_block.append_operation(GPUOperation(GPUOpType.CONSTANT, results=[c_start], attributes={"value": start}))
                offs_val = GPUValue(self.new_var_name("offs"), I32)
                gpu_block.append_operation(GPUOperation(GPUOpType.ADD, operands=[tid_val, c_start], results=[offs_val]))
                self.val_map[hl_res] = offs_val
            else:
                self.val_map[hl_res] = tid_val

        elif op.op_type in {HLOpType.ADD, HLOpType.SUB, HLOpType.MUL, HLOpType.DIV}:
            lhs = self.val_map[op.operands[0]]
            rhs = self.val_map[op.operands[1]]
            hl_res = op.results[0]
            res_type = hl_res.type.element_type if isinstance(hl_res.type, TensorType) else hl_res.type
            res_val = GPUValue(self.new_var_name("v"), res_type)
            self.val_map[hl_res] = res_val

            gpu_op_map = {
                HLOpType.ADD: GPUOpType.ADD,
                HLOpType.SUB: GPUOpType.SUB,
                HLOpType.MUL: GPUOpType.MUL,
                HLOpType.DIV: GPUOpType.DIV,
            }
            gpu_block.append_operation(
                GPUOperation(gpu_op_map[op.op_type], operands=[lhs, rhs], results=[res_val])
            )

        elif op.op_type == HLOpType.CMP:
            lhs = self.val_map[op.operands[0]]
            rhs = self.val_map[op.operands[1]]
            hl_res = op.results[0]
            res_val = GPUValue(self.new_var_name("cond"), I1)
            self.val_map[hl_res] = res_val

            gpu_block.append_operation(
                GPUOperation(
                    GPUOpType.CMP,
                    operands=[lhs, rhs],
                    results=[res_val],
                    attributes={"predicate": op.attributes["predicate"]},
                )
            )

        elif op.op_type == HLOpType.WHERE:
            cond = self.val_map[op.operands[0]]
            t_val = self.val_map[op.operands[1]]
            f_val = self.val_map[op.operands[2]]
            hl_res = op.results[0]
            res_type = hl_res.type.element_type if isinstance(hl_res.type, TensorType) else hl_res.type
            res_val = GPUValue(self.new_var_name("sel"), res_type)
            self.val_map[hl_res] = res_val

            gpu_block.append_operation(
                GPUOperation(GPUOpType.SELECT, operands=[cond, t_val, f_val], results=[res_val])
            )

        elif op.op_type == HLOpType.LOAD:
            ptr = self.val_map[op.operands[0]]
            offs = self.val_map[op.operands[1]]
            mask = self.val_map[op.operands[2]] if len(op.operands) > 2 else None
            
            hl_res = op.results[0]
            res_type = hl_res.type.element_type if isinstance(hl_res.type, TensorType) else hl_res.type
            res_val = GPUValue(self.new_var_name("ld"), res_type)
            self.val_map[hl_res] = res_val

            operands = [ptr, offs]
            if mask:
                operands.append(mask)

            gpu_block.append_operation(
                GPUOperation(
                    GPUOpType.GLOBAL_LOAD,
                    operands=operands,
                    results=[res_val],
                    attributes={"has_mask": mask is not None},
                )
            )

        elif op.op_type == HLOpType.STORE:
            ptr = self.val_map[op.operands[0]]
            offs = self.val_map[op.operands[1]]
            val = self.val_map[op.operands[2]]
            mask = self.val_map[op.operands[3]] if len(op.operands) > 3 else None

            # Check if this function is a 2D matrix kernel (has DOT operation)
            has_dot = any(o.op_type == HLOpType.DOT for b in hl_func.blocks for o in b.operations)
            if has_dot:
                # Resolve dimension arguments M and N dynamically from non-pointer scalar parameters
                scalar_args = [self.val_map[arg] for arg in hl_func.args if not isinstance(arg.type, PointerType)]
                var_m = scalar_args[0] if len(scalar_args) > 0 else GPUValue("var_M", I32)
                var_n = scalar_args[1] if len(scalar_args) > 1 else GPUValue("var_N", I32)

                gpu_block.append_operation(
                    GPUOperation(
                        GPUOpType.TILED_STORE,
                        operands=[ptr, val, var_m, var_n],
                        results=[],
                    )
                )
            else:
                operands = [ptr, offs, val]
                if mask:
                    operands.append(mask)

                gpu_block.append_operation(
                    GPUOperation(
                        GPUOpType.GLOBAL_STORE,
                        operands=operands,
                        results=[],
                        attributes={"has_mask": mask is not None},
                    )
                )

        elif op.op_type in {HLOpType.REDUCE_SUM, HLOpType.REDUCE_MAX}:
            in_val = self.val_map[op.operands[0]]
            hl_res = op.results[0]
            res_val = GPUValue(self.new_var_name("red"), hl_res.type)
            self.val_map[hl_res] = res_val
            red_op = GPUOpType.REDUCE_SUM if op.op_type == HLOpType.REDUCE_SUM else GPUOpType.REDUCE_MAX

            gpu_block.append_operation(
                GPUOperation(red_op, operands=[in_val], results=[res_val])
            )

        elif op.op_type == HLOpType.DOT:
            a_val = self.val_map[op.operands[0]]
            b_val = self.val_map[op.operands[1]]
            hl_res = op.results[0]
            res_val = GPUValue(self.new_var_name("tiled_dot"), hl_res.type)
            self.val_map[hl_res] = res_val

            # Trace DOT operands (a_val, b_val) back through SSA LOAD operations to get true matrix buffer pointers
            def trace_load_ptr(val_target: HLValue) -> Optional[GPUValue]:
                for b in hl_func.blocks:
                    for o in b.operations:
                        if o.op_type == HLOpType.LOAD and len(o.results) > 0:
                            if o.results[0] == val_target:
                                return self.val_map[o.operands[0]]
                # Handle block arguments (e.g. from loop header)
                for b in hl_func.blocks:
                    if val_target in b.args:
                        arg_i = b.args.index(val_target)
                        for pred_b in hl_func.blocks:
                            for o in pred_b.operations:
                                if o.op_type == HLOpType.BR and len(o.operands) > arg_i:
                                    res_ptr = trace_load_ptr(o.operands[arg_i])
                                    if res_ptr:
                                        return res_ptr
                return None

            traced_a = trace_load_ptr(op.operands[0])
            traced_b = trace_load_ptr(op.operands[1])

            if traced_a is None or traced_b is None:
                raise RuntimeError(
                    f"Backend Lowering Error: Unable to trace matrix source pointers for tf.dot operands {op.operands[0]} and {op.operands[1]}"
                )

            ptr_a = traced_a
            ptr_b = traced_b

            # Dynamically derive M, N, K scalar parameters from function signature non-pointer arguments
            scalar_args = [self.val_map[arg] for arg in hl_func.args if not isinstance(arg.type, PointerType)]
            m_val = scalar_args[0] if len(scalar_args) > 0 else None
            n_val = scalar_args[1] if len(scalar_args) > 1 else None
            k_val = scalar_args[2] if len(scalar_args) > 2 else None

            bm = op.attributes.get("BM", 16)
            bn = op.attributes.get("BN", 16)
            bk = op.attributes.get("BK", 16)

            dot_operands = [ptr_a, ptr_b]
            if m_val: dot_operands.append(m_val)
            if n_val: dot_operands.append(n_val)
            if k_val: dot_operands.append(k_val)

            gpu_block.append_operation(
                GPUOperation(
                    GPUOpType.TILED_DOT,
                    operands=dot_operands,
                    results=[res_val],
                    attributes={"BM": bm, "BN": bn, "BK": bk},
                )
            )

        elif op.op_type == HLOpType.BR:
            target_block = self.block_map[op.successors[0]]
            gpu_args = [self.val_map[o] for o in op.operands]
            gpu_block.append_operation(
                GPUOperation(GPUOpType.BR, operands=gpu_args, results=[], successors=[target_block])
            )

        elif op.op_type == HLOpType.COND_BR:
            cond_val = self.val_map[op.operands[0]]
            then_block = self.block_map[op.successors[0]]
            else_block = self.block_map[op.successors[1]]

            t_count = op.attributes.get("then_arg_count", 0)
            e_count = op.attributes.get("else_arg_count", 0)

            t_args = [self.val_map[o] for o in op.operands[1:1 + t_count]]
            e_args = [self.val_map[o] for o in op.operands[1 + t_count:1 + t_count + e_count]]

            gpu_block.append_operation(
                GPUOperation(
                    GPUOpType.COND_BR,
                    operands=[cond_val] + t_args + e_args,
                    results=[],
                    attributes={"then_arg_count": len(t_args), "else_arg_count": len(e_args)},
                    successors=[then_block, else_block],
                )
            )

        elif op.op_type == HLOpType.RETURN:
            gpu_block.append_operation(
                GPUOperation(GPUOpType.RETURN, operands=[], results=[])
            )
