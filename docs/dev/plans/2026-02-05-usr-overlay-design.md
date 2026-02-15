# USR Overlay-First Design

Date: 2026-02-05
Owner: Eric J. South

## Summary
USR will be overlay-first. The base dataset lives in `records.parquet` and is treated as immutable canonical data. Derived columns are stored in `_derived/*.parquet` overlays keyed by explicit join keys. Reads merge overlays by default, and a new `usr materialize` command is the explicit way to fold overlays into the base. No back-compat behavior is preserved. All semantics are strict and explicit with no silent fallbacks.

## Goals
- Reduce I/O amplification and memory use for large datasets.
- Make behavior deterministic and assertive.
- Decouple tool outputs from base data.
- Make the schema contract explicit, enforceable, and stable.
- Ensure remote sync verification is explicit and auditable.

## Non-Goals
- Backward compatibility with legacy implicit behaviors.
- Hidden defaults or best-effort fallbacks.

## Architecture
- Base table: `records.parquet` with required core fields only.
- Overlays: `_derived/<tool>.parquet` with a join key and namespaced columns.
- Reads (head, describe, export) merge overlays by default into a unified view.
- Writes to base use streaming Parquet rewrites with atomic rename.
- Writes to overlays are streaming and avoid touching the base.
- `usr materialize` folds overlays into the base and clears overlays as an explicit action.

## CLI And Library Contract
- Library-first: CLI is a thin Typer adapter.
- Explicit join keys only: `attach` and `dedupe` require `--key`.
- Strict attach semantics:
  - Attachment keys must be unique.
  - Missing values error unless `--allow-missing`.
  - Derived column collisions error unless `--overwrite`.
  - All derived columns must be namespaced as `<tool>__<field>`.
- Dedupe semantics:
  - Explicit key required.
  - Case-insensitive keys only via `sequence_ci` for `dna_4` and must be explicit.
  - `keep-first` and `keep-last` supported with streaming logic.
- Remotes config:
  - `USR_REMOTES_PATH` is required for commands that use remotes.
- Remote verification:
  - `--verify {auto,hash,size,parquet}` required.
  - Explicit warnings on downgrade; no silent fallback.

## Data Flow And Concurrency
- All mutations acquire a dataset-local write lock (`.usr.lock`).
- Reads are lock-free.
- Base rewrite uses ParquetWriter to temp file then atomic rename.
- Overlay writes are atomic and isolated per tool.
- Event logging happens only after successful commit.

## Deterministic ID And Schema Contract
- ID derivation: `id = sha1(utf8(f"{bio_type}{DELIM}{sequence_norm}"))`.
- `bio_type` must be in an explicit allow-list and cannot contain `DELIM`.
- `sequence_norm = sequence.strip()` and `length == len(sequence_norm)`.
- Required columns have explicit types and nullability.
- Namespacing rules are enforced with allowed character regex and reserved namespaces.

## Events
- `.events.log` becomes JSONL with fields:
  - timestamp_utc (RFC3339)
  - action
  - args summary
  - fingerprint (rows, cols, file size, optional sha256 via env flag)
  - tool version
- Events are appended only after a successful file commit.

## Testing
- TDD for all behavior changes.
- Unit tests:
  - ID determinism and delimiter enforcement.
  - Validation on large datasets using streaming uniqueness.
  - Attach enforcement of explicit keys, namespacing, missing, overwrite policies.
  - Overlay creation and merge semantics.
  - Remotes config requiring `USR_REMOTES_PATH`.
  - Verify mode behavior with explicit downgrade warnings.
- Integration test:
  - Create dataset, import rows, attach overlay, validate, materialize.
  - Confirm JSONL event entries with fingerprints.

## Migration Strategy
- No backward compatibility. Existing datasets are treated as base-only.
- Overlays become the default for any new derived outputs.

## Open Questions
- None. Decisions are final for implementation.
