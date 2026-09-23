"""Command Line Interface (CLI) for TileForge compiler."""

import argparse
import importlib.util
import sys
from pathlib import Path

from tileforge.driver import Compiler
from tileforge.ir.types import F32, I32, PointerType


def main() -> None:
    parser = argparse.ArgumentParser(prog="tileforge", description="TileForge Tensor-Kernel Compiler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # tileforge compile <file.py>
    compile_cmd = subparsers.add_parser("compile", help="Compile kernel file and print IR stages")
    compile_cmd.add_argument("filename", type=str, help="Python source file containing TileForge kernel")
    compile_cmd.add_argument("--backend", type=str, default="cpu", choices=["cpu", "metal"], help="Target backend (cpu or metal)")
    compile_cmd.add_argument("--show-cfg", action="store_true", help="Display CFG block layout")
    compile_cmd.add_argument("--emit-cfg", type=str, default=None, help="Save Graphviz DOT CFG output to file")
    compile_cmd.add_argument("--emit-metal", type=str, default=None, help="Save generated MSL source code to file")

    # tileforge run <file.py>
    run_cmd = subparsers.add_parser("run", help="Compile and execute kernel file")
    run_cmd.add_argument("filename", type=str, help="Python source file to execute")
    run_cmd.add_argument("--backend", type=str, default="cpu", choices=["cpu", "metal"], help="Target backend (cpu or metal)")

    args = parser.parse_args()
    filepath = Path(args.filename)

    if not filepath.exists():
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.command == "compile":
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        compiler = Compiler(optimize=True, backend=args.backend)
        default_args = [PointerType(F32), PointerType(F32), PointerType(F32), I32]
        try:
            res = compiler.compile(code, default_args)
            print("==========================================")
            print(f"TILEFORGE KERNEL: {res.kernel_name} [{args.backend.upper()} TARGET]")
            print("==========================================")
            print("\n--- UNOPTIMIZED IR ---")
            print(res.ir_before_optimization)
            print("\n--- OPTIMIZED IR ---")
            print(res.ir_after_optimization)

            if res.backend_ir:
                print("\n--- GPU BACKEND IR ---")
                print(res.backend_ir)

            if res.generated_source:
                print("\n--- GENERATED METAL SHADING LANGUAGE (MSL) ---")
                print(res.generated_source)

            if args.show_cfg:
                print("\n--- CFG BASIC BLOCKS ---")
                print(" -> ".join(res.cfg))

            if args.emit_cfg:
                with open(args.emit_cfg, "w", encoding="utf-8") as f:
                    f.write(res.dot)
                print(f"\n✓ Saved CFG DOT file to '{args.emit_cfg}'")

            if args.emit_metal and res.generated_source:
                with open(args.emit_metal, "w", encoding="utf-8") as f:
                    f.write(res.generated_source)
                print(f"\n✓ Saved MSL source file to '{args.emit_metal}'")

            print("==========================================")
        except Exception as e:
            print(f"Compilation Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "run":
        spec = importlib.util.spec_from_file_location("user_kernel_mod", filepath)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load python module from '{filepath}'", file=sys.stderr)
            sys.exit(1)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


if __name__ == "__main__":
    main()
