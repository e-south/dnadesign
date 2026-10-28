# Artifact Manifests

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28

Each persisted artifact directory contains a `manifest.json` with at least:

- schema version
- artifact kind and id
- workspace id
- command provenance
- recorded inputs and digests
- primary outputs
- row and dimension stats where relevant

Reuse checks cover three independent conditions: the recorded inputs and
configuration still match, the declared output payloads have their recorded
SHA-256 digests, and the producing implementation matches the current tool.
A changed or missing payload, missing digest evidence, or implementation mismatch
reports `attention` and causes recipes to rebuild that artifact.

Manifests record `execution_identity` from the installed LatentDNA, SeqLedger and
shared-contract Python sources plus the versions of NumPy, PyArrow, SciPy,
scikit-learn, UMAP and Pydantic. This conservatively invalidates artifacts after
implementation changes. External producers remain responsible for their own
code and environment records; their supplied files are tracked as inputs.

Managed dependencies record an upstream manifest digest and recursively verify
its declared outputs. Hashes are streamed without loading arrays and cached by
file path, size, inode, modification time and change time. One freshness traversal
shares resolved digests across dependent artifacts. Source overlays may use
explicit inventory and digest-ledger contracts to avoid rescanning their full
external trees.

Rebuilds write to staging directories and publish completed payloads and manifests
together. A failed computation or installation retains the previous artifact.
Use `--force` to replace an existing artifact after changing its configuration.
Notebook generation validates the staged notebook before replacing an existing
notebook; a first-time failed generation retains diagnostics with `status: error`.

See also:

- [deliverable-contract.md](deliverable-contract.md)
- [performance-budgets.md](performance-budgets.md)
