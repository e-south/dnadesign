# Thread Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-24

`dnadesign.thread` is intentionally small right now. Its first public surface is
the generic ProteinMPNN adapter under
`dnadesign.thread.adapters.proteinmpnn`.

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

Study packages own biological masks, evidence interpretation, source
selection, candidate ids, fold-check policy, and candidate-ranking policy.
