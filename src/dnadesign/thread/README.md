![thread banner](assets/thread-banner.svg)

`dnadesign.thread` owns generic fixed-backbone request, sample-ingest,
candidate-normalization, and fold-check artifact mechanics that are independent
of a study's biological policy.

## Documentation

- [Thread docs index](docs/README.md): current public surfaces and boundaries.
- [Repository docs index](../../../docs/README.md): repo-wide workflow routing.

## Current Surface

- `dnadesign.thread.adapters.proteinmpnn` builds and validates
  helper-compatible ProteinMPNN request artifacts and normalizes official
  ProteinMPNN outputs into sample rows.
- `dnadesign.thread.candidates` converts accepted backend samples into stable
  candidate rows with canonical mutation summaries and mask-audit fields.
- `dnadesign.thread.foldcheck` defines model-agnostic fold-check request and
  report contracts. It records sequence ids, WT baseline coverage, runtime
  provenance, thresholds, and pass/fail rows without choosing a fold backend.
- It owns chain-local position conversion, helper JSONL payloads,
  protein-only backbone export, request manifests, request hashes, explicit
  official-checkout preflight, helper parity checks, backend-run manifests,
  sample tables, candidate tables, fold-check manifests, and generic
  no-fallback validation.
- It does not choose study masks, select biological priors, run fold models, or
  create study handoff decisions.

Study packages should keep biological choices local, then call `thread` only
for reusable fixed-backbone mechanics.
