# USR dataset layout and code map

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


## Dataset layout

```text
src/dnadesign/usr/
├─ src/
├─ datasets/
│  ├─ <namespace>/
│  │  └─ <dataset_name>/
│  │     ├─ records.parquet
│  │     ├─ _derived/
│  │     ├─ meta.md
│  │     ├─ .events.log
│  │     ├─ _registry/
│  │     └─ _snapshots/
│  └─ _archive/
│     └─ <namespace>/<dataset_name>/...
└─ demo_material/
```

Dataset ids should be namespace-qualified (`namespace/dataset`).

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
