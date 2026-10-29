[Infer guide](../docs/README.md)

# Workspaces and inputs

A workspace keeps a job's configuration, input files and output directories together. You can create one anywhere you have write access.

```bash
# Inspect available packaged workspaces.
infer workspace list

# Create a workspace with one local JSONL input.
infer workspace init --id my_inference --profile local
```

The default destination is `./workspaces/my_inference`. Set `--root` to choose another location, or set `INFER_WORKSPACE_ROOT` as your default. An existing workspace is never overwritten.

## What is created

```text
workspaces/my_inference/
├── config.yaml
├── inputs/
│   └── records.jsonl
└── outputs/
    └── logs/ops/audit/
```

The `local` profile supplies a small input and an extraction configuration for validation. Follow the [quickstart](../docs/getting-started/cli-quickstart.md) before configuring a model runtime.

The `usr-pressure` profile instead prepares a SeqLedger extraction configuration and an `outputs/usr_datasets/` directory. It does **not** create or populate the dataset. The [SeqLedger procedure](../docs/operations/pressure-test-agnostic-models.md) covers that step.

## Supply your input

Each job declares an `ingest` section in `config.yaml`:

| `ingest.source` | Input | Required location |
| --- | --- | --- |
| `sequences` | One sequence per nonempty text line | `path`, such as `inputs/sequences.txt` |
| `records` | One JSON object per line, with a sequence field | `path` and `field` |
| `pt_file` | A supported PyTorch data file | Optional `path`, otherwise `<job_id>.pt` |
| `usr` | A SeqLedger dataset | `dataset` and a dataset root |

Relative paths resolve from the configuration file's directory. The schema uses `source: usr` for SeqLedger, while CLI options use `--seqledger`.

For example, to read your JSONL records:

```yaml
# Read the sequence field from each JSONL record.
ingest:
  source: records
  path: inputs/records.jsonl
  field: sequence
```

For SeqLedger writes, set `ingest.root` explicitly. Use a workspace-local root for a self-contained job, or the shared root that owns the dataset. See the [command reference](../docs/reference/command-contracts.md) for read-only root resolution and PyTorch input requirements.

## Feature-bundle example

The [Evo2 bundle example](evo2_feature_bundle_smoke/README.md) includes short sequences with and without flanking context. It is a separate packaged example, not a `workspace init` profile.

## Reset stored extraction outputs

```bash
# Archive the infer overlay for one dataset before recomputing its columns.
infer prune --seqledger <dataset-id> --seqledger-root <dataset-root>
```

This affects the `infer` overlay only. It preserves the base records, workspace files and sequence-view feature sidecars. See [feature storage](../docs/reference/feature-schema.md) for sidecar inspection and compaction.
