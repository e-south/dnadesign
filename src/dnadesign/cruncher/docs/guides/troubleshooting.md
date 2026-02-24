## Troubleshooting

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Missing lockfile](#missing-lockfile)
- [Targets not ready](#targets-not-ready)
- [MEME tools unavailable](#meme-tools-unavailable)
- [Analyze missing artifacts](#analyze-missing-artifacts)
- [HT datasets return zero rows](#ht-datasets-return-zero-rows)
- [Portfolio prepare failures](#portfolio-prepare-failures)

### Missing lockfile
Symptom:
- `Missing lockfile ... Run cruncher lock first.`

Checks:
- Verify `workspaces/<name>/.cruncher/locks/<config>.lock.json` exists.
- Confirm source preference matches discovered source IDs.

Fix:
```bash
uv run cruncher lock -c configs/config.yaml
```

### Targets not ready
Symptom:
- sample/analyze refuses to run due to unresolved targets.

Checks:
```bash
uv run cruncher targets status -c configs/config.yaml
```

Fix:
- Run missing preflight stages from the workspace runbook:
```bash
uv run cruncher workspaces run --runbook configs/runbook.yaml --step lock_targets --step parse_run
```

### MEME tools unavailable
Symptom:
- `discover motifs` fails with tool resolution errors.

Checks:
```bash
uv run cruncher discover check -c configs/config.yaml
command -v meme streme fimo
```

Fix:
- Ensure `discover.tool_path` points to the MEME bin directory or export PATH accordingly.

### Analyze missing artifacts
Symptom:
- `analyze` fails on missing `optimize/tables/*.parquet` or trace/state files.

Checks:
- Inspect run entrypoint files from [`../reference/artifacts.md`](../reference/artifacts.md).

Fix:
```bash
uv run cruncher sample --force-overwrite -c configs/config.yaml
uv run cruncher analyze --summary -c configs/config.yaml
```

### HT datasets return zero rows
Symptom:
- fetch succeeds but HT rows are empty for `tfbinding` mode.

Checks:
```bash
uv run cruncher sources datasets regulondb -c configs/config.yaml --tf <TF_NAME>
```

Fix:
- Set `ingest.regulondb.ht_binding_mode: peaks`.
- Provide hydration inputs (`ingest.genome_fasta` or `--genome-fasta`) if sequences are not returned.

### Portfolio prepare failures
Symptom:
- portfolio run fails during source preparation or source artifact validation.

Checks:
- Verify each source has `prepare.runbook: configs/runbook.yaml`.
- Verify required step IDs exist in each source runbook.

Fix:
```bash
uv run cruncher portfolio run --spec configs/portfolios/<name>.portfolio.yaml --prepare-ready rerun --force-overwrite
```
