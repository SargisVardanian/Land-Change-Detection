# Static analysis policy

The SigLIP2 track is checked with the canonical cluster Python for compile/tests and the installed Pyright executable for type analysis.

Project-owned contracts are not globally suppressed. The required diagnostics remain errors:

- reportRedeclaration
- reportOptionalMemberAccess
- reportArgumentType
- reportGeneralTypeIssues

The new package is included explicitly as src/qcpr_siglip2. Missing-import and private-import suppressions remain limited to the existing repository policy because the cluster environment provides optional runtime packages through the system installation. PyTorch runtime narrowing is performed at explicit module boundaries; deserialized manifest values are validated before use. No model or dataset path is embedded in Python source.
