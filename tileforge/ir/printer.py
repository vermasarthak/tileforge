"""Deterministic textual IR printer for TileForge IR."""

from __future__ import annotations
from tileforge.ir.module import Module
from tileforge.ir.function import Function
from tileforge.ir.block import Block
from tileforge.ir.operation import Operation, OpType


class IRPrinter:
    """Prints TileForge IR in human-readable, SSA-deterministic text format."""
    def __init__(self, indent: str = "  "):
        self.indent: str = indent

    def print_module(self, module: Module) -> str:
        lines = []
        for func in module.functions:
            lines.append(self.print_function(func))
        return "\n\n".join(lines)

    def print_function(self, func: Function) -> str:
        lines = []
        args_str = ",\n".join(f"{self.indent * 2}{arg.name}: {arg.type}" for arg in func.args)
        if args_str:
            lines.append(f"func @{func.name}(\n{args_str}\n) {{")
        else:
            lines.append(f"func @{func.name}() {{")

        for block in func.blocks:
            lines.append(self.print_block(block))
        
        lines.append("}")
        return "\n".join(lines)

    def print_block(self, block: Block) -> str:
        lines = []
        if block.name != "entry":
            lines.append(f"^{block.name}:")
        
        for op in block.operations:
            lines.append(f"{self.indent}{self.print_operation(op)}")
        return "\n".join(lines)

    def print_operation(self, op: Operation) -> str:
        res_str = op.results[0].name if op.results else None
        res_type_str = f" : {op.results[0].type}" if op.results else ""

        if op.op_type == OpType.CONSTANT:
            val = op.attributes.get("value")
            return f"{res_str} = tf.constant {val}{res_type_str}"

        elif op.op_type == OpType.PROGRAM_ID:
            axis = op.attributes.get("axis", 0)
            return f"{res_str} = tf.program_id axis={axis}{res_type_str}"

        elif op.op_type == OpType.ARANGE:
            start = op.attributes.get("start", 0)
            end = op.attributes.get("end", 0)
            return f"{res_str} = tf.arange {start}, {end}{res_type_str}"

        elif op.op_type == OpType.CMP:
            pred = op.attributes.get("predicate", "eq")
            lhs = op.operands[0].name
            rhs = op.operands[1].name
            return f"{res_str} = tf.cmp {pred} {lhs}, {rhs}{res_type_str}"

        elif op.op_type in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}:
            op_name = op.op_type
            lhs = op.operands[0].name
            rhs = op.operands[1].name
            return f"{res_str} = {op_name} {lhs}, {rhs}{res_type_str}"

        elif op.op_type == OpType.LOAD:
            ptr = op.operands[0].name
            offs = op.operands[1].name
            mask_str = f", mask={op.operands[2].name}" if len(op.operands) > 2 else ""
            return f"{res_str} = tf.load {ptr}[{offs}]{mask_str}{res_type_str}"

        elif op.op_type == OpType.STORE:
            ptr = op.operands[0].name
            offs = op.operands[1].name
            val = op.operands[2].name
            mask_str = f", mask={op.operands[3].name}" if len(op.operands) > 3 else ""
            return f"tf.store {ptr}[{offs}], {val}{mask_str}"

        elif op.op_type == OpType.WHERE:
            cond = op.operands[0].name
            t_val = op.operands[1].name
            f_val = op.operands[2].name
            return f"{res_str} = tf.where {cond}, {t_val}, {f_val}{res_type_str}"

        elif op.op_type == OpType.RETURN:
            if op.operands:
                return f"tf.return {op.operands[0].name}"
            return "tf.return"

        else:
            opnds = ", ".join(o.name for o in op.operands)
            res_part = f"{res_str} = " if res_str else ""
            return f"{res_part}{op.op_type} {opnds}{res_type_str}"
