<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Infer guide](../README.md) / [Reference](README.md)

# Command reference

Use `infer <command> --help` for the full option list. The examples assume an activated [Infer environment](../getting-started/cli-quickstart.md#1-install-and-activate-infer).

## Run jobs

`infer run` accepts a YAML configuration or a preset, but not both:

```bash
# Check a configured job and resolve its inputs without loading weights.
infer run --config config.yaml --dry-run

# Validate a preset against an existing SeqLedger dataset.
infer run --preset evo2/extract_logits_ll \
  --seqledger my_sequences --seqledger-root ./datasets --dry-run
```

In configuration mode, `--job <id>` selects jobs from the file. Selecting no matching jobs fails. Input selection and write-back settings belong in YAML; `--seqledger`, `--field`, `--ids`, `--seqledger-root` and `--write-back` are preset-mode options.

### Input configuration

| `ingest.source` | Location and behavior |
| --- | --- |
| `sequences` | `path` to a text file with one sequence per nonempty line |
| `records` | `path` to JSONL objects and `field` naming the sequence field |
| `pt_file` | Optional `path`, defaulting to `<config-dir>/<job_id>.pt` |
| `usr` | SeqLedger `dataset`, `field` and root; `path` is not accepted |

Relative paths resolve from the configuration file's directory. The SeqLedger schema value is **`usr`**, not `seqledger`.

SeqLedger reads require `ingest.root` or `DNADESIGN_SEQLEDGER_ROOT`. Jobs that write back must set `ingest.root` explicitly. Configuration-mode dry runs check the resolved dataset and required columns as well as local file inputs.

The CLI requires `--i-know-this-is-pickle` or `INFER_ALLOW_PICKLE=1` for `pt_file` input. Loading still uses PyTorch's restricted `weights_only=True` mode; the flag does not enable arbitrary object deserialization. Prefer text or JSONL for new sequence inputs.

## Extract or generate directly

`infer extract` accepts a preset or an ad-hoc function and output format. It can read inline sequences, record files, PyTorch files or a SeqLedger dataset. `--write-back` requires a compatible storage destination.

```bash
infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run
```

`infer generate` accepts a prompt and generation settings. Successful runtime output contains `gen_seqs`; malformed adapter output is rejected. See the [quickstart generation example](../getting-started/cli-quickstart.md#4-try-an-ad-hoc-request).

Local CLI extraction displays an output summary. To retain results, configure SeqLedger write-back or use the [Python API](../dev/README.md) to receive the returned values.

## Inspect models and presets

```bash
infer adapters list
infer adapters fns
infer presets list
infer presets show evo2/extract_logits_ll
```

`adapters list` reports registered model IDs, not proof that their dependencies or weights are installed. `adapters fns` lists function contracts. `adapters cache-clear` clears the runtime adapter cache.

## Validate inputs and outputs

- `validate config` checks the model/job schema and runtime configuration. Supply `--config`, or run beside a `config.yaml`. Unknown keys and invalid types produce exit code `2`.
- `validate seqledger` checks access to a dataset and sequence field.
- `validate seqledger-registry --config config.yaml` derives the output namespace and type registration required for write-back.
- `validate sequence-view-completion --config config.yaml` checks reusable, missing and stale feature payloads. See [feature storage](feature-schema.md).

## Workspaces and stored results

`workspace where` resolves the workspace root in this order: `--root`, `INFER_WORKSPACE_ROOT`, then `./workspaces`. `workspace init` accepts `local` or `usr-pressure` profiles, or an explicit `--template`. IDs must be directory names; existing workspaces are never overwritten. The [workspace guide](../../workspaces/README.md) describes the files created.

`prune` archives the Infer overlay by default. It preserves base records and sequence-view sidecars. `features migrate` and `features compact` maintain sidecar payload storage; their scope and integrity checks are in the [feature schema](feature-schema.md#payload-storage-and-maintenance).

## Write and resume behavior

Output cardinality mismatches, unreadable resume data and unsupported write-back destinations fail the operation. With `overwrite: false`, an existing Infer overlay value cannot be replaced for the same record and column. Feature bundles also use request metadata to determine whether stored results are reusable. See the [Evo2 resume contract](evo2-provider.md#resume-and-digest-behavior).
