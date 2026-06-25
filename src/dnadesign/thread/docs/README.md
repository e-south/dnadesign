# Thread Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-24

`dnadesign.thread` is intentionally small right now. Its public surfaces are the
generic ProteinMPNN adapter, ProteinMPNN candidate normalization, and generic
fold-check request/report contracts.

The adapter owns reusable fixed-backbone mechanics:

- chain-local ProteinMPNN position conversion
- helper-compatible JSONL payloads
- protein-only backbone export
- request manifests and request hashes
- explicit official-checkout preflight
- helper parity checks before backend execution
- backend-run manifests
- normalized sample tables
- generic no-fallback request validation

The fold-check contract owns model-agnostic artifact shape:

- fold-check FASTA request records
- WT baseline presence
- runtime kind/version and parameter hash fields
- threshold id and threshold values
- normalized fold-check report rows with accepted/rejected/errored states

Study packages own biological masks, evidence interpretation, source selection,
candidate batch policy, fold-check threshold policy, and candidate-ranking
policy.
