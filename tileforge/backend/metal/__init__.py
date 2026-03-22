"""Metal target backend package init."""

from tileforge.backend.metal.compiler import MetalBackend, MetalCompiledKernel
from tileforge.backend.metal.ir import GPUModule, GPUFunction, GPUBlock, GPUOperation, GPUOpType
from tileforge.backend.metal.printer import GPUIRPrinter
from tileforge.backend.metal.codegen import MSLCodeGenerator

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
