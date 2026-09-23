# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in TileForge, please report it privately:

- **Email**: `sarthakverma0802@gmail.com`
- **Subject**: `[SECURITY] TileForge Vulnerability Report`

Please include:
1. Description of the issue (e.g. AST parsing denial-of-service, buffer overflow in ctypes Metal runtime, or unvalidated host code execution).
2. Minimal reproducible test case.
3. Impact assessment.

## Security Model & Sandboxing

- TileForge parses restricted Python functions using Python's `ast.parse` and statically validates allowed syntax nodes before evaluation.
- Arbitrary Python code execution (dynamic `eval`, `exec`, arbitrary module imports, class declarations) is rejected by the AST verifier.
- The GPU runtime allocates unified memory buffers through Apple's Metal Objective-C APIs via `ctypes`. Buffer sizes and bounds should be validated before launching untrusted kernel pipelines.
