"""Command Line Interface (CLI) for TileForge compiler."""

import sys
import argparse
import importlib.util
from pathlib import Path

from tileforge.frontend.parser import Parser
from tileforge.frontend.semantic import SemanticAnalyzer
from tileforge.lowering.ast_to_ir import ASTToLowering
from tileforge.ir.types import PointerType, F32, I32
from tileforge.driver import Compiler


def main() -> None:
    parser = argparse.ArgumentParser(prog="tileforge", description="TileForge Tensor-Kernel Compiler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # tileforge compile <file.py>
    compile_cmd = subparsers.add_parser("compile", help="Compile kernel file and print IR stages")
    compile_cmd.add_argument("filename", type=str, help="Python source file containing TileForge kernel")

    # tileforge run <file.py>
    run_cmd = subparsers.add_parser("run", help="Compile and execute kernel file via CPU reference interpreter")
    run_cmd.add_argument("filename", type=str, help="Python source file to execute")

    args = parser.parse_args()
    filepath = Path(args.filename)

    if not filepath.exists():
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.command == "compile":
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        compiler = Compiler(optimize=True)
        # Default argument types for vector_add style kernels
        default_args = [PointerType(F32), PointerType(F32), PointerType(F32), I32]
        try:
            res = compiler.compile(code, default_args)
            print("==========================================")
            print(f"TILEFORGE KERNEL: {res.kernel_name}")
            print("==========================================")
            print("\n--- UNOPTIMIZED IR ---")
            print(res.ir_before_optimization)
            print("\n--- OPTIMIZED IR ---")
            print(res.ir_after_optimization)
            print("==========================================")
        except Exception as e:
            print(f"Compilation Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "run":
        # Import file as python module and run it
        spec = importlib.util.spec_from_file_location("user_kernel_mod", filepath)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load python module from '{filepath}'", file=sys.stderr)
            sys.exit(1)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


if __name__ == "__main__":
    main()
