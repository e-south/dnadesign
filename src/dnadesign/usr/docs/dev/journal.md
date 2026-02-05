# USR Refactoring Journal

## 2026-02-05
- Audit focus: streaming + dedup for large datasets; densegen/USR integration; notify integration patterns.
- Observations:
  - Dataset operations (`import_rows`, `attach`, `get`, `grep`, `export`, `merge`, `dedupe`) read the full `records.parquet` into memory and rewrite the full file. This preserves atomicity but scales poorly for large datasets and for frequent updates (e.g., densegen resume).
  - `USRWriter` (densegen) caches existing ids but attaches metadata per flush. With small `chunk_size` this rewrites the dataset many times; larger chunks or deferring attach would reduce I/O.
  - Remote config search paths do not include `src/dnadesign/usr/remotes.yaml`, but docs/AGENTS point there. This is a doc/code mismatch.
  - PyArrow sysctl warnings appear in non-TTY environments unless `USR_SUPPRESS_PYARROW_SYSCTL=1` is set.
- Next: propose minimal UX improvements and doc alignment; consider optional streaming writer or id index for large datasets.

- Decision: dataset layout should be namespaced (no active/archive buckets). CLI resolution will accept unqualified names only when unambiguous; otherwise require `namespace/dataset` for clarity.
- Decision: remotes config should be decoupled and explicit. Add `USR_REMOTES_PATH` override and fail fast if multiple config files are found.
- Update: added column projection support for `head`/`export` and optimized `get` to scan via pyarrow.dataset. Added tests for read ops.
- Update: grep now supports batch_size for large datasets (streamed scan with early cutoff).
- Update: CLI entrypoint uses Typer only; argparse adapter removed.
- Update: convert_legacy lazy-loads torch; CLI import does not pull torch.
- Update: remotes config defaults to repo-local remotes.yaml with USR_REMOTES_PATH override.
- Update: describe/head/export stream Parquet in batches; info/schema use metadata only.
- Update: CLI meta notes and dataset selection are strict; no silent fallbacks.
- Update: remote Parquet stats require python + pyarrow on remote; sysctl filter is explicit opt-in.
- Update: datasets index README tracked; .gitignore allows the index file.
- Decision: move to overlay-first datasets. Derived columns live in `_derived/*.parquet`; base `records.parquet` holds core fields only; `usr materialize` explicitly merges overlays.
- Decision: explicit join keys only for `attach` and `dedupe`; no default key.
- Decision: `USR_REMOTES_PATH` is required for remotes; no implicit defaults.
- Decision: reads merge overlays by default; base-only output requires explicit flag.
- Decision: dataset writes use a write lock only; reads are lock-free.
- Decision: no backward compatibility or silent fallbacks.
- Update: remotes config now requires `USR_REMOTES_PATH`; repo-local fallback removed.
