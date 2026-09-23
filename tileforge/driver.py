"""Public Compiler Driver API for TileForge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from tileforge.frontend.parser import Parser
from tileforge.frontend.semantic import SemanticAnalyzer
from tileforge.ir.dot_exporter import export_cfg_dot
from tileforge.ir.function import Function
from tileforge.ir.module import Module
from tileforge.ir.printer import IRPrinter
from tileforge.ir.types import Type
from tileforge.language.decorators import KernelFunction
from tileforge.lowering.ast_to_ir import ASTToLowering
from tileforge.passes.algebraic import AlgebraicSimplifyPass
from tileforge.passes.constant_fold import ConstantFoldPass
from tileforge.passes.cse import CSEPass
from tileforge.passes.dce import DeadCodeEliminationPass
from tileforge.passes.manager import PassManager
from tileforge.passes.simplify_cfg import SimplifyCFGPass
from tileforge.runtime.interpreter import CPUInterpreter


@dataclass
class CompilationResult:
    """Holds compilation artifacts and provides launch capabilities."""
    kernel_name: str
    module: Module
    function: Function
    ir_before_optimization: str
    ir_after_optimization: str
    interpreter: CPUInterpreter
    backend_name: str = "cpu"
    backend_ir: Optional[str] = None
    generated_source: Optional[str] = None
    compiled_kernel: Any = None

    @property
    def cfg(self) -> List[str]:
        return [b.name for b in self.function.blocks]

    @property
    def dot(self) -> str:
        return export_cfg_dot(self.function)

    def launch(self, grid: Tuple[int, ...], args: List[Any]) -> None:
        """Launches kernel execution on specified target backend."""
        if self.backend_name == "metal" and self.compiled_kernel is not None:
            self.compiled_kernel.launch(grid=grid, args=args)
        else:
            self.interpreter.execute(self.function, grid=grid, args=args)


class Compiler:
    """TileForge Compiler Driver executing full frontend, IR lowering, optimization, and target execution."""
    def __init__(self, optimize: bool = True, backend: str = "cpu"):
        if backend not in {"cpu", "metal"}:
            raise ValueError(f"Unsupported compiler backend '{backend}'. Must be 'cpu' or 'metal'")
        self.optimize: bool = optimize
        self.backend_name: str = backend
        self.printer: IRPrinter = IRPrinter()
        self.interpreter: CPUInterpreter = CPUInterpreter()

    def compile(
        self,
        kernel: Union[KernelFunction, Callable[..., Any], str],
        arg_types: List[Type],
    ) -> CompilationResult:
        # 1. Frontend: Parse source to TileForge AST
        parser = Parser()
        kernel_ast = parser.parse_kernel(kernel)

        # 2. Semantic Analysis: Symbol & Type check
        analyzer = SemanticAnalyzer()
        analyzer.analyze(kernel_ast, arg_types)

        # 3. Lowering: AST -> SSA IR
        lowering = ASTToLowering()
        func = lowering.lower_kernel(kernel_ast, arg_types)
        ir_before = self.printer.print_module(lowering.module)

        # 4. Optimization Passes
        if self.optimize:
            pm = PassManager([
                SimplifyCFGPass(),
                ConstantFoldPass(),
                AlgebraicSimplifyPass(),
                CSEPass(),
                DeadCodeEliminationPass(),
                SimplifyCFGPass(),
            ], verify_each=True)
            pm.run(lowering.module)

        ir_after = self.printer.print_module(lowering.module)

        backend_ir_str: Optional[str] = None
        gen_src: Optional[str] = None
        compiled_k: Any = None

        if self.backend_name == "metal":
            from tileforge.backend.metal import MetalBackend
            metal_backend = MetalBackend()
            gpu_mod = metal_backend.lower(lowering.module)
            compiled_k = metal_backend.compile(gpu_mod, kernel_ast.name)
            backend_ir_str = metal_backend.printer.print_module(gpu_mod)
            gen_src = compiled_k.msl_source

        return CompilationResult(
            kernel_name=kernel_ast.name,
            module=lowering.module,
            function=func,
            ir_before_optimization=ir_before,
            ir_after_optimization=ir_after,
            interpreter=self.interpreter,
            backend_name=self.backend_name,
            backend_ir=backend_ir_str,
            generated_source=gen_src,
            compiled_kernel=compiled_k,
        )
