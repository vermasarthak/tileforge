# Contributing to TileForge

TileForge welcomes contributions to compiler passes, AST lowerings, type system expansions, and backend code generators.

## Prerequisites
- Python 3.11+
- Apple Silicon Mac with Metal support (for Metal GPU execution tests) or Linux (for CPU reference interpreter and compiler IR tests)

## Local Development Workflow

1. **Environment Setup**:
   ```bash
   git clone https://github.com/vermasarthak/tileforge.git
   cd tileforge
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install pytest numpy ruff
   ```

2. **Linting & Code Style**:
   ```bash
   ruff check .
   ```

3. **Running Tests**:
   ```bash
   # Run all non-Metal tests (Linux / macOS)
   pytest -v -k "not metal and not gpu"

   # Run full test suite including Apple Metal GPU execution (macOS Apple Silicon)
   pytest -v
   ```

4. **Running Benchmarks**:
   ```bash
   python benchmarks/run_benchmarks.py
   python benchmarks/run_matmul_benchmark.py
   ```

## Contribution Invariants
- **Property-Preserving Transformations**: Any new optimization pass must include property tests proving output equivalence against the NumPy CPU reference interpreter across random seeds.
- **Strict SSA Verification**: Any IR transformation must pass `IRVerifier.verify_module()` without undominated Value usages.
