"""Deterministic Metal Shading Language (MSL) Generator.

Generates readable C++14 MSL with:
- 1D vector kernels (Vector Add, Reductions)
- 2D Threadgroup-Tiled GEMM kernels (shared memory tileA/tileB, barriers, edge masking)
- C++ switch state-machine for CFG/loops
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
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
from tileforge.ir.types import Type, PointerType, TensorType, PrimitiveType, I32, I1, F32, I64


def sanitize_name(name: str) -> str:
    if name.startswith('%'): return 'var_' + name[1:]
    if name.startswith('^'): return 'block_' + name[1:]
    return name


class MSLCodeGenerator:
    """Generates C++ Metal Shading Language (MSL) source code from GPU Backend IR."""

    def __init__(self):
        self.type_map: Dict[Type, str] = {I32: "int", I1: "bool", F32: "float", I64: "long"}

    def format_type(self, t: Type) -> str:
        if isinstance(t, PrimitiveType): return self.type_map.get(t, "float")
        elif isinstance(t, PointerType): return f"device {self.format_type(t.element_type)}*"
        elif isinstance(t, TensorType): return self.format_type(t.element_type)
        return "float"

    def generate(self, gpu_module: GPUModule) -> str:
        lines = ["#include <metal_stdlib>", "using namespace metal;", ""]
        for func in gpu_module.functions:
            lines.append(self.generate_function(func))
        return "\n".join(lines)

    def generate_function(self, func: GPUFunction) -> str:
        # Detect if function is a 2D threadgroup tiled GEMM kernel
        has_tiled_dot = any(
            op.op_type in {GPUOpType.TILED_DOT, GPUOpType.DOT}
            for block in func.blocks
            for op in block.operations
        ) or func.is_2d_grid

        if has_tiled_dot:
            return self.generate_tiled_matmul_function(func)

        return self.generate_standard_function(func)

    def generate_standard_function(self, func: GPUFunction) -> str:
        param_lines = []
        for i, arg in enumerate(func.args):
            if isinstance(arg.type, PointerType):
                param_lines.append(f"    device {self.format_type(arg.type.element_type)}* {sanitize_name(arg.name)} [[buffer({i})]]")
            else:
                param_lines.append(f"    constant {self.format_type(arg.type)}& {sanitize_name(arg.name)} [[buffer({i})]]")
        param_lines.append("    uint3 tid [[thread_position_in_threadgroup]]")
        param_lines.append("    uint3 tgid [[threadgroup_position_in_grid]]")

        code_lines = [f"kernel void {sanitize_name(func.name)}(", ",\n".join(param_lines), ") {"]

        has_reduction = any(op.op_type in {GPUOpType.REDUCE_SUM, GPUOpType.REDUCE_MAX} for block in func.blocks for op in block.operations)
        if has_reduction: code_lines.append("    threadgroup float shared_scratch[256];")

        # Declare variables
        declared_vars = {}
        for block in func.blocks:
            for arg in block.args: declared_vars[sanitize_name(arg.name)] = arg.type
            for op in block.operations:
                for res in op.results: declared_vars[sanitize_name(res.name)] = res.type

        for name, typ in declared_vars.items():
            code_lines.append(f"    {self.format_type(typ)} {name};")

        # Assign integer IDs to blocks
        block_ids = {block.name: i for i, block in enumerate(func.blocks)}

        if not func.blocks:
            code_lines.append("}")
            return "\n".join(code_lines)

        code_lines.append(f"    int _state = {block_ids[func.blocks[0].name]};")
        code_lines.append("    while (true) {")
        code_lines.append("        switch (_state) {")

        for block in func.blocks:
            code_lines.append(f"            case {block_ids[block.name]}: {{")
            for op in block.operations:
                stmt = self.generate_operation(op, block_ids)
                if stmt:
                    for line in stmt.split('\n'):
                        code_lines.append(f"                {line}")
            code_lines.append("            }")

        code_lines.append("        }")
        code_lines.append("    }")
        code_lines.append("}")
        return "\n".join(code_lines)

    def generate_tiled_matmul_function(self, func: GPUFunction) -> str:
        param_lines = []
        for i, arg in enumerate(func.args):
            if isinstance(arg.type, PointerType):
                param_lines.append(f"    device const {self.format_type(arg.type.element_type)}* {sanitize_name(arg.name)} [[buffer({i})]]" if i < 2 else f"    device {self.format_type(arg.type.element_type)}* {sanitize_name(arg.name)} [[buffer({i})]]")
            else:
                param_lines.append(f"    constant int& {sanitize_name(arg.name)} [[buffer({i})]]")
        param_lines.append("    uint2 tid [[thread_position_in_threadgroup]]")
        param_lines.append("    uint2 tgid [[threadgroup_position_in_grid]]")

        # Param names assume (A, B, C, M, N, K) signature
        arg_names = [sanitize_name(a.name) for a in func.args]
        var_A = arg_names[0] if len(arg_names) > 0 else "A"
        var_B = arg_names[1] if len(arg_names) > 1 else "B"
        var_C = arg_names[2] if len(arg_names) > 2 else "C"
        var_M = arg_names[3] if len(arg_names) > 3 else "M"
        var_N = arg_names[4] if len(arg_names) > 4 else "N"
        var_K = arg_names[5] if len(arg_names) > 5 else "K"

        msl = f"""kernel void {sanitize_name(func.name)}(
{",\n".join(param_lines)}
) {{
    // 2D Threadgroup shared memory tile allocation (BM=16, BN=16, BK=16)
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];

    // 2D Thread and Block indexing
    uint tid_x = tid.x; // column local (0..15)
    uint tid_y = tid.y; // row local (0..15)

    uint row = tgid.y * 16 + tid_y;
    uint col = tgid.x * 16 + tid_x;

    float acc = 0.0f;

    // K-tile loop over global K dimension
    for (int k0 = 0; k0 < {var_K}; k0 += 16) {{
        // Cooperative load tileA [BM x BK] with edge masking
        if (row < (uint){var_M} && (k0 + tid_x) < (uint){var_K}) {{
            tileA[tid_y][tid_x] = {var_A}[row * {var_K} + (k0 + tid_x)];
        }} else {{
            tileA[tid_y][tid_x] = 0.0f;
        }}

        // Cooperative load tileB [BK x BN] with edge masking
        if ((k0 + tid_y) < (uint){var_K} && col < (uint){var_N}) {{
            tileB[tid_y][tid_x] = {var_B}[(k0 + tid_y) * {var_N} + col];
        }} else {{
            tileB[tid_y][tid_x] = 0.0f;
        }}

        // Barrier 1: Wait for threadgroup tile load to complete
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 16x16 tile dot product into local accumulator
        for (int k_inner = 0; k_inner < 16; ++k_inner) {{
            acc += tileA[tid_y][k_inner] * tileB[k_inner][tid_x];
        }}

        // Barrier 2: Wait for tile compute to finish before overwriting shared memory in next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write back computed accumulator to global memory with edge masking
    if (row < (uint){var_M} && col < (uint){var_N}) {{
        {var_C}[row * {var_N} + col] = acc;
    }}
}}"""
        return msl

    def generate_operation(self, op: GPUOperation, block_ids: Dict[str, int]) -> str:
        if op.op_type == GPUOpType.PROGRAM_ID:
            axis = op.attributes.get("axis", 0)
            axis_attr = ["tgid.x", "tgid.y", "tgid.z"][axis]
            return f"{sanitize_name(op.results[0].name)} = (int){axis_attr};"
        elif op.op_type == GPUOpType.THREAD_ID:
            return f"{sanitize_name(op.results[0].name)} = (int)tid.x;"
        elif op.op_type == GPUOpType.THREAD_ID_X:
            return f"{sanitize_name(op.results[0].name)} = (int)tid.x;"
        elif op.op_type == GPUOpType.THREAD_ID_Y:
            return f"{sanitize_name(op.results[0].name)} = (int)tid.y;"
        elif op.op_type == GPUOpType.THREADGROUP_ID_X:
            return f"{sanitize_name(op.results[0].name)} = (int)tgid.x;"
        elif op.op_type == GPUOpType.THREADGROUP_ID_Y:
            return f"{sanitize_name(op.results[0].name)} = (int)tgid.y;"
        elif op.op_type == GPUOpType.CONSTANT:
            val = op.attributes["value"]
            val_repr = str(val) if not isinstance(val, bool) else ("true" if val else "false")
            if isinstance(val, float) and "." not in val_repr and "e" not in val_repr: val_repr += ".0f"
            elif isinstance(val, float): val_repr += "f"
            return f"{sanitize_name(op.results[0].name)} = {val_repr};"
        elif op.op_type in {GPUOpType.ADD, GPUOpType.SUB, GPUOpType.MUL, GPUOpType.DIV}:
            sym = {GPUOpType.ADD: "+", GPUOpType.SUB: "-", GPUOpType.MUL: "*", GPUOpType.DIV: "/"}[op.op_type]
            return f"{sanitize_name(op.results[0].name)} = {sanitize_name(op.operands[0].name)} {sym} {sanitize_name(op.operands[1].name)};"
        elif op.op_type == GPUOpType.CMP:
            sym = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}[op.attributes["predicate"]]
            return f"{sanitize_name(op.results[0].name)} = ({sanitize_name(op.operands[0].name)} {sym} {sanitize_name(op.operands[1].name)});"
        elif op.op_type == GPUOpType.SELECT:
            return f"{sanitize_name(op.results[0].name)} = {sanitize_name(op.operands[0].name)} ? {sanitize_name(op.operands[1].name)} : {sanitize_name(op.operands[2].name)};"
        elif op.op_type == GPUOpType.GLOBAL_LOAD:
            ptr = sanitize_name(op.operands[0].name)
            offs = sanitize_name(op.operands[1].name)
            if op.attributes.get("has_mask", False) and len(op.operands) > 2:
                return f"{sanitize_name(op.results[0].name)} = {sanitize_name(op.operands[2].name)} ? {ptr}[{offs}] : 0.0f;"
            return f"{sanitize_name(op.results[0].name)} = {ptr}[{offs}];"
        elif op.op_type == GPUOpType.GLOBAL_STORE:
            ptr = sanitize_name(op.operands[0].name)
            offs = sanitize_name(op.operands[1].name)
            val = sanitize_name(op.operands[2].name)
            if op.attributes.get("has_mask", False) and len(op.operands) > 3:
                return f"if ({sanitize_name(op.operands[3].name)}) {{ {ptr}[{offs}] = {val}; }}"
            return f"{ptr}[{offs}] = {val};"
        elif op.op_type in {GPUOpType.REDUCE_SUM, GPUOpType.REDUCE_MAX}:
            val = sanitize_name(op.operands[0].name)
            accum_expr = f"shared_scratch[tid.x] += shared_scratch[tid.x + s];" if op.op_type == GPUOpType.REDUCE_SUM else f"shared_scratch[tid.x] = max(shared_scratch[tid.x], shared_scratch[tid.x + s]);"
            return (
                f"shared_scratch[tid.x] = {val};\n"
                f"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                f"for (uint s = 128; s > 0; s >>= 1) {{\n"
                f"    if (tid.x < s) {{\n"
                f"        {accum_expr}\n"
                f"    }}\n"
                f"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                f"}}\n"
                f"{sanitize_name(op.results[0].name)} = shared_scratch[0];"
            )
        elif op.op_type == GPUOpType.BARRIER:
            return "threadgroup_barrier(mem_flags::mem_threadgroup);"
        elif op.op_type == GPUOpType.DOT:
            return f"{sanitize_name(op.results[0].name)} = {sanitize_name(op.operands[0].name)} * {sanitize_name(op.operands[1].name)};"
        elif op.op_type == GPUOpType.BR:
            target = op.successors[0]
            lines = [f"{sanitize_name(block_arg.name)} = {sanitize_name(arg_val.name)};" for arg_val, block_arg in zip(op.operands, target.args)]
            lines.append(f"_state = {block_ids[target.name]};")
            lines.append("break;")
            return "\n".join(lines)
        elif op.op_type == GPUOpType.COND_BR:
            cond = sanitize_name(op.operands[0].name)
            then_target = op.successors[0]
            else_target = op.successors[1]
            t_count = op.attributes.get("then_arg_count", 0)
            t_args = op.operands[1:1 + t_count]
            e_args = op.operands[1 + t_count:]
            lines = [f"if ({cond}) {{"]
            for arg_val, block_arg in zip(t_args, then_target.args): lines.append(f"    {sanitize_name(block_arg.name)} = {sanitize_name(arg_val.name)};")
            lines.append(f"    _state = {block_ids[then_target.name]};")
            lines.append("} else {")
            for arg_val, block_arg in zip(e_args, else_target.args): lines.append(f"    {sanitize_name(block_arg.name)} = {sanitize_name(arg_val.name)};")
            lines.append(f"    _state = {block_ids[else_target.name]};")
            lines.append("}")
            lines.append("break;")
            return "\n".join(lines)
        elif op.op_type == GPUOpType.RETURN:
            return "return;"
        return ""
