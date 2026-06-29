---
doc_id: dnadesign-thread-docs
surface: tool-docs
owner: dnadesign-maintainers
last_verified: 2026-06-29
---

# Thread Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-29

`dnadesign.thread` is intentionally small right now. Its public surfaces are the
generic ProteinMPNN adapter, generic ColabFold output normalizer, generic ESM
Atlas annotation adapter, ProteinMPNN candidate normalization, generic
fold-check request/report contracts, and a generic structure-prediction
registry. It also exposes a small browser structure-view contract for
notebook-based review of existing PDB/mmCIF files.

The adapter owns reusable fixed-backbone mechanics:

- chain-local ProteinMPNN position conversion
- helper-compatible parsed-PDB, assigned-chain, and fixed-position JSONL payloads
- protein-only backbone export
- request manifests and request hashes
- explicit official-checkout preflight
- helper parity checks before backend execution
- backend-run manifests
- normalized sample tables
- generic no-fallback request validation

The adapter follows the public ProteinMPNN CLI path. It prepares sidecars,
checks them against official helpers, runs `protein_mpnn_run.py` only when an
explicit checkout root is provided, and normalizes the resulting sequences. It
does not decide study masks, infer raw PDB residue ids as fixed positions, or
silently fall back to another inverse-folding backend.

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

The adapter normalizes completed `colabfold_batch` output directories. It does
not install ColabFold, call a hosted API, submit scheduler jobs, or choose the
study's fold-check thresholds.

`dnadesign.thread.adapters.esm_atlas` owns reusable Atlas API annotation
mechanics:

- no-auth Atlas protein lookup by sequence MD5 hash
- bounded query parameters and explicit `fold_on_miss` handling
- query MD5 hashes, request hashes, and raw response hashes
- top SAE feature summaries
- sparse protein-level activations
- sparse per-residue activations
- compact feature catalog rows
- explicit error rows when the alpha API drifts or a query is absent
- optional Atlas on-demand structure payload extraction when a caller has
  explicitly enabled `fold_on_miss`

The adapter does not claim function, rank processivity, or choose which study
candidates to query. Study packages own sequence selection, feature panel
interpretation, WT-relative comparisons, and assay-panel decisions. Do not add a
wider semantic-profile framework until a second backend needs the same contract.

`dnadesign.thread.adapters.biohub_esmc` owns authenticated Biohub ESMC
query-time SAE mechanics:

- runtime-only credential loading with redacted manifests
- public `POST /api/v1/encode` -> `POST /api/v1/logits` request flow
- encoded SAE tensor decoding
- sparse per-sequence and per-residue SAE feature rows
- explicit error rows for API, decode, or schema failures

This is not an Atlas lookup adapter and not a fold adapter. Study wrappers may
use it to annotate synthetic sequences that do not exist in Atlas, but those
rows remain semantic context unless a study-specific policy uses them later.

`dnadesign.thread.structure_predictions` owns the backend-neutral registry for
model-predicted structures. It records candidate id, sequence hash, backend,
model family/name/version, runtime or endpoint, parameter hash, request hash,
raw response hash, structure hash, structure source URI, local structure path,
confidence fields, status, and failure reason. This registry is deliberately
separate from fold-check reports and Atlas semantic profiles. A ColabFold
structure used for fold validation and an Atlas/ESMFold-derived structure for
the same sequence must be two provenance-separated rows, not one merged result.

`dnadesign.thread.structure_views` owns browser-embedded structure-view
contracts for existing structure files. It defines backend-neutral model/view
specs and currently renders HTML through py3Dmol. This package is for
interactive notebook review of structures that already exist; it does not run a
fold model, pick study structures, capture publication camera poses, or replace
ChimeraX still renders.

Study packages own biological masks, evidence interpretation, source selection,
candidate batch policy, fold-check threshold policy, and candidate-ranking
policy.

Fold model execution is a runtime boundary, not a hidden `thread` fallback.
Scheduler templates, device storage, and environment activation belong to the
operator surface that runs the model. `thread` should parse or write compact
fold-check artifacts through public contracts; it should not choose a study's
fold backend, thresholds, or downstream promotion rule.
