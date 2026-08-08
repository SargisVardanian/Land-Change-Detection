# Static analysis policy

The SigLIP2 track is checked with the canonical cluster Python for compile/tests and the installed Pyright executable for type analysis. The current cluster image does not provide Ruff, Flake8, Pyflakes, or Pylint; this is recorded as an environment limitation rather than silently reported as a lint pass.

Project-owned contracts are not globally suppressed. The required diagnostics remain errors:

- reportRedeclaration
- reportOptionalMemberAccess
- reportArgumentType
- reportGeneralTypeIssues

The typed Model-v3 scope is explicitly `src/qcpr_temporal_siglip`; the older
`src/qcpr_siglip2` backend remains covered by compileall and the full pytest
suite but is not silently presented as a clean new typed package. Missing-import
diagnostics remain errors. The only suppression is scoped to the new package's
`reportPrivateImportUsage`, because the cluster Pyright build does not export
runtime Torch symbols such as `torch.cat` and `torch.float32` in its stubs. It
does not suppress redeclaration, optional access, argument type, or general
type diagnostics. PyTorch runtime narrowing is performed at explicit module
boundaries; deserialized manifest values are validated before use. No model or
dataset path is embedded in Python source.

Current final gates: full pytest `670 passed, 3 skipped, 16 warnings`; compileall PASS; Pyright `0 errors, 0 warnings`; shell syntax PASS; `git diff --check` PASS; Ruff unavailable on the cluster image.
