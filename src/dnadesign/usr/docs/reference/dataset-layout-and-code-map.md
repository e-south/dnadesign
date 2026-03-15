# USR dataset layout and code map

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14


## Dataset layout

```text
src/dnadesign/usr/
├─ src/
├─ datasets/
│  ├─ <dataset_id>/
│  │  ├─ records.parquet
│  │  ├─ _derived/
│  │  ├─ meta.md
│  │  ├─ .events.log
│  │  ├─ _registry/
│  │  └─ _snapshots/
│  ├─ <namespace>/<dataset_name>/...
│  └─ _archive/
│     └─ <dataset_id-or-qualified-path>/...
└─ demo_material/
```

Dataset ids may be flat (`dataset`) or namespace-qualified (`namespace/dataset`).

Choose the least-coupled semantic id that still keeps the dataset understandable.

- Prefer a flat dataset id when the biological collection is already specific, such as `mg1655_promoters`, `plasmids`, or `pdual10_slot_a_window_1kb_demo`.
- Use namespace-qualified ids only when they genuinely improve disambiguation instead of encoding tool routing.
- Keep tool provenance in namespaced overlay columns such as `construct__*`, `densegen__*`, or `infer__*`, not in the dataset id itself, unless the dataset is truly tool-private scratch state.
- Carry human-readable record names in record columns such as `usr_label__primary` / `usr_label__aliases`, not only in local notes or workspace conventions.

Legacy dataset ids under `archived/` are rejected with hard errors, including path-first commands targeting `datasets/archived/**` and `usr/archived/**`.

## Maintainer code map

Core dataset orchestration:

- `src/dnadesign/usr/src/dataset.py`
- `src/dnadesign/usr/src/dataset_activity.py`
- `src/dnadesign/usr/src/dataset_materialize.py`
- `src/dnadesign/usr/src/dataset_overlay_ops.py`
- `src/dnadesign/usr/src/dataset_registry_modes.py`
- `src/dnadesign/usr/src/dataset_dedupe.py`

CLI decomposition:

- `src/dnadesign/usr/src/cli.py`
- `src/dnadesign/usr/src/cli_commands/read.py`
- `src/dnadesign/usr/src/cli_commands/write.py`
- `src/dnadesign/usr/src/cli_commands/state.py`
- `src/dnadesign/usr/src/cli_commands/sync.py`
- `src/dnadesign/usr/src/cli_commands/remotes.py`

## Next steps

- Schema details: [schema-contract.md](schema-contract.md)
- Overlay and registry semantics: [overlay-and-registry.md](overlay-and-registry.md)
