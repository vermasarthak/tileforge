"""Public Compiler Driver API for TileForge."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Union

from tileforge.frontend.parser import Parser
from tileforge.frontend.semantic import SemanticAnalyzer
from tileforge.lowering.ast_to_ir import ASTToLowering
from tileforge.ir.types import Type
from tileforge.ir.module import Module
from tileforge.ir.function import Function
from tileforge.ir.printer import IRPrinter
from tileforge.passes.manager import PassManager
from tileforge.passes.constant_fold import ConstantFoldPass
from tileforge.passes.algebraic import AlgebraicSimplifyPass
from tileforge.passes.dce import DeadCodeEliminationPass
from tileforge.passes.cse import CSEPass
from tileforge.runtime.interpreter import CPUInterpreter
from tileforge.language.decorators import KernelFunction


@dataclass
class CompilationResult:
    """Holds compilation artifacts and provides launch capabilities."""
    kernel_name: str
    module: Module
    function: Function
    ir_before_optimization: str
    ir_after_optimization: str
    interpreter: CPUInterpreter

    def launch(self, grid: Tuple[int, ...], args: List[Any]) -> None:
        """Launches kernel execution on CPU reference interpreter."""
        self.interpreter.execute(self.function, grid=grid, args=args)


class Compiler:
    """TileForge Compiler Driver executing full frontend, IR lowering, optimization, and target execution."""
    def __init__(self, optimize: bool = True):
        self.optimize: bool = optimize
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
                ConstantFoldPass(),
                AlgebraicSimplifyPass(),
                CSEPass(),
                DeadCodeEliminationPass(),
            ], verify_each=True)
            pm.run(lowering.module)

        ir_after = self.printer.print_module(lowering.module)

        return CompilationResult(
            kernel_name=kernel_ast.name,
            module=lowering.module,
            function=func,
            ir_before_optimization=ir_before,
            ir_after_optimization=ir_after,
            interpreter=self.interpreter,
        )
