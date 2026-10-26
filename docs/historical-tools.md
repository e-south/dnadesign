## Historical tools

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-26

LibShuffle, Billboard, NMF and TFKDAnalysis are retired from the current package.
Their source and tracked examples remain available at revision
[`1e550a690`](https://github.com/e-south/dnadesign/tree/1e550a690af9de98b7bdde6d7f5e6dc00659cb2c).
Restore that revision in a separate checkout when reproducing older work.
Local results not stored in Git were preserved in the maintainer's verified
software archive before removal.

DenseGen is being replaced for new construction work by
[DenseArrays](https://github.com/e-south/dense-arrays). Existing DenseGen
workflows and pinned scientific environments remain available during migration.
Use the exact source and dependency pins recorded by a study for historical replay.

Current choices are listed in the [tool catalog](../README.md#available-tools).

The study-specific USR PyTorch maintenance script was retired at revision
[`3235d502d`](https://github.com/e-south/dnadesign/tree/3235d502d5f24af386b1ee3a7a1bf2fc445766fc/src/dnadesign/usr/scripts/archived_pytorch_manager.py).
Its source and the former ignored prototype shelves are preserved in verified
external recovery archives, including modified files and nested Git histories.
They are not installed runtime components. USR's supported legacy-data conversion
commands remain available.

Permuter's placeholder evaluator was removed from the installed runtime.
Deterministic scoring fixtures live only in tests. Existing generated scores are
not rewritten; new evaluations require an explicitly configured evaluator.
