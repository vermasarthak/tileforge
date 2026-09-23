"""Pass infrastructure for TileForge IR optimizations."""

from __future__ import annotations

from typing import List

from tileforge.ir.module import Module
from tileforge.ir.verifier import IRVerifier


class Pass:
    """Base class for all TileForge optimization passes."""
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, module: Module) -> bool:
        """Runs optimization pass on the module. Returns True if module was modified."""
        raise NotImplementedError


class PassManager:
    """Executes a series of optimization passes with strict IR verification."""
    def __init__(self, passes: List[Pass], verify_each: bool = True):
        self.passes: List[Pass] = passes
        self.verify_each: bool = verify_each
        self.verifier: IRVerifier = IRVerifier()

    def run(self, module: Module) -> bool:
        if self.verify_each:
            self.verifier.verify_module(module)

        modified_any = False
        for pass_ in self.passes:
            changed = pass_.run(module)
            if changed:
                modified_any = True
                if self.verify_each:
                    try:
                        self.verifier.verify_module(module)
                    except Exception as e:
                        from tileforge.ir.printer import IRPrinter
                        print(f"[PassManager Error after {pass_.__class__.__name__}]: {e}")
                        print(IRPrinter().print_module(module))
                        raise e

        return modified_any
