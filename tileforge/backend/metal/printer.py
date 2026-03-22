"""Deterministic Printer for GPU Backend IR."""

from __future__ import annotations
from tileforge.backend.metal.ir import GPUModule, GPUFunction, GPUBlock, GPUOperation, GPUKernelArg


class GPUIRPrinter:
    """Prints GPU Backend IR into readable deterministic string format."""

    def print_module(self, module: GPUModule) -> str:
        lines: list[str] = []
        for func in module.functions:
            lines.append(self.print_function(func))
        return "\n\n".join(lines)

    def print_function(self, func: GPUFunction) -> str:
        args_formatted = ", ".join(f"%{a.name}: {a.address_space.value} {a.type}" for a in func.args)
        lines = [f"gpu.kernel @{func.name}({args_formatted}) {{"]

        for block in func.blocks:
            args_str = f"({', '.join('%' + a.name + ': ' + str(a.type) for a in block.args)})" if block.args else ""
            lines.append(f"^{block.name}{args_str}:")
            for op in block.operations:
                lines.append(f"  {self.print_operation(op)}")

        lines.append("}")
        return "\n".join(lines)

    def print_operation(self, op: GPUOperation) -> str:
        res_str = ", ".join(f"%{r.name}" for r in op.results)
        if res_str:
            res_str += " = "

        opnd_str = ", ".join(f"%{o.name}" for o in op.operands)
        attr_str = ""
        if op.attributes:
            attrs_formatted = [f"{k}={v}" for k, v in sorted(op.attributes.items())]
            attr_str = " " + " ".join(attrs_formatted)

        succ_str = ""
        if op.successors:
            succ_str = " [" + ", ".join(f"^{s.name}" for s in op.successors) + "]"

        type_str = ""
        if op.results:
            type_str = f" : {op.results[0].type}"

        return f"{res_str}{op.op_type.value} {opnd_str}{attr_str}{succ_str}{type_str}"
