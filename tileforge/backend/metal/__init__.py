"""Metal target backend package init."""

from tileforge.backend.metal.codegen import MSLCodeGenerator
from tileforge.backend.metal.compiler import MetalBackend, MetalCompiledKernel
from tileforge.backend.metal.ir import GPUBlock, GPUFunction, GPUModule, GPUOperation, GPUOpType
from tileforge.backend.metal.printer import GPUIRPrinter

__all__ = [
    "MetalBackend",
    "MetalCompiledKernel",
    "GPUModule",
    "GPUFunction",
    "GPUBlock",
    "GPUOperation",
    "GPUOpType",
    "GPUIRPrinter",
    "MSLCodeGenerator",
]
