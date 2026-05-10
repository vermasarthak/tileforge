"""Metal Backend Compiler and Execution Pipeline Manager with Caching."""

from __future__ import annotations
import hashlib
import os
from typing import Any, Dict, List, Tuple, Optional
from tileforge.backend.base import Backend, CompiledKernel
from tileforge.ir.module import Module as HLModule
from tileforge.backend.metal.ir import GPUModule
from tileforge.backend.metal.lowering import HLToGPULowering
from tileforge.backend.metal.verifier import GPUIRVerifier
from tileforge.backend.metal.printer import GPUIRPrinter
from tileforge.backend.metal.codegen import MSLCodeGenerator
from tileforge.backend.metal.runtime import MetalRuntime


class MetalCompiledKernel(CompiledKernel):
    """Executes a compiled Metal pipeline state with deterministic caching."""

    def __init__(
        self,
        kernel_name: str,
        msl_source: str,
        gpu_module: GPUModule,
        pipeline_state: int,
        runtime: MetalRuntime,
        threadgroup_dimensions: Tuple[int, ...] = (256, 1, 1),
    ):
        self.kernel_name = kernel_name
        self.msl_source = msl_source
        self.gpu_module = gpu_module
        self.pipeline_state = pipeline_state
        self.runtime = runtime
        self.threadgroup_dimensions = threadgroup_dimensions

    def launch(self, grid: Tuple[int, ...], args: List[Any]) -> None:
        self.runtime.dispatch(self.pipeline_state, grid, args, threads_per_threadgroup=self.threadgroup_dimensions)


class MetalBackend(Backend):
    """Native Apple Metal GPU Target Backend."""

    def __init__(self):
        self._runtime: Optional[MetalRuntime] = None
        self.lowering_pass = HLToGPULowering()
        self.verifier = GPUIRVerifier()
        self.printer = GPUIRPrinter()
        self.codegen = MSLCodeGenerator()
        self.pipeline_cache: Dict[str, Tuple[MetalCompiledKernel, str, str]] = {}

    @property
    def name(self) -> str:
        return "metal"

    @property
    def runtime(self) -> MetalRuntime:
        if self._runtime is None:
            self._runtime = MetalRuntime()
        return self._runtime

    def lower(self, module: HLModule) -> GPUModule:
        gpu_mod = self.lowering_pass.lower_module(module)
        self.verifier.verify_module(gpu_mod)
        return gpu_mod

    def compile(self, lowered_module: GPUModule, func_name: str) -> MetalCompiledKernel:
        msl_source = self.codegen.generate(lowered_module)
        if os.environ.get("TILEFORGE_DEBUG") == "1":
            print("\n--- DEBUG: GENERATED MSL SOURCE ---\n" + msl_source + "\n-------------------------------------")

        # Find function metadata for threadgroup dimensions
        tg_dims = (256, 1, 1)
        for f in lowered_module.functions:
            if f.name == func_name:
                tg_dims = f.threadgroup_dimensions
                break

        # Compute deterministic cache key
        cache_key = hashlib.sha256((msl_source + str(tg_dims)).encode("utf-8")).hexdigest()
        if cache_key in self.pipeline_cache:
            compiled, _, _ = self.pipeline_cache[cache_key]
            return compiled

        pipeline = self.runtime.compiler.compile_source(msl_source, func_name)
        compiled_kernel = MetalCompiledKernel(
            kernel_name=func_name,
            msl_source=msl_source,
            gpu_module=lowered_module,
            pipeline_state=pipeline,
            runtime=self.runtime,
            threadgroup_dimensions=tg_dims,
        )

        gpu_ir_str = self.printer.print_module(lowered_module)
        self.pipeline_cache[cache_key] = (compiled_kernel, gpu_ir_str, msl_source)
        return compiled_kernel
