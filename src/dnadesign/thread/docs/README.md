---
doc_id: dnadesign-thread-docs
surface: tool-docs
owner: dnadesign-maintainers
last_verified: 2026-06-25
---

# Thread Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-25

`dnadesign.thread` is intentionally small right now. Its public surfaces are the
generic ProteinMPNN adapter, generic ColabFold output normalizer,
ProteinMPNN candidate normalization, and generic fold-check request/report
contracts.

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
- request-sequence hash binding
- runtime kind/version and parameter hash fields
- threshold id and threshold values
- normalized fold-check report rows with accepted/rejected/errored states
- subset FASTA and external-run manifest preparation through
  `python -m dnadesign.thread.foldcheck.subset`

`dnadesign.thread.adapters.colabfold` owns reusable ColabFold output parsing:

- ColabFold output-file discovery by request sequence id, with longer
  manifest sequence ids matched before shorter prefixes and file matching
  limited to exact ids or known ColabFold-generated suffixes
- one-pass output indexing with rank-token parsing for ColabFold model/score
  files
- pLDDT extraction from model PDB B-factors
- PAE JSON summarization when available
- C-alpha RMSD against the WT runtime baseline or an explicit reference PDB
- failure rows for missing output or missing required metrics

Study packages own biological masks, evidence interpretation, source selection,
candidate batch policy, fold-check threshold policy, and candidate-ranking
policy.

Fold model execution is a runtime boundary, not a hidden `thread` fallback.
Scheduler templates, device storage, and environment activation belong to the
operator surface that runs the model. `thread` should parse or write compact
fold-check artifacts through public contracts; it should not choose a study's
fold backend, thresholds, or downstream promotion rule.
