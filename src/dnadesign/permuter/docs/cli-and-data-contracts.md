# Permuter CLI And Data Contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-24

Permuter mutates biological sequences, scores variants with pluggable
evaluators, and writes one scoped dataset per workspace run.

## Source Layout

```text
src/dnadesign/permuter/
  __init__.py                 # public API facade and CLI entrypoint
  src/                        # implementation internals
    resources/                # packaged reusable resources
  workspaces/
    _shared/inputs/           # packaged shared reference inputs
    <scope>/config.yaml       # one runnable scope
    <scope>/outputs/          # generated, ignored
  docs/
  tests/
```

There is no root-level `api.py`, `cli.py`, `jobs/`, `inputs/`, `results/`, or
`notebooks/` surface. New runnable units are workspace scopes, not loose root
configs.

## Workspace Contract

Each scope is a directory with a single `config.yaml`. The directory name must
match `scope.name`, and `output.dir` must resolve inside the workspace root.
Supported output layouts are `flat` and `nested`; retired layout names are
contract errors rather than compatibility aliases.

```yaml
scope:
  name: rnaseh1_nt_scan
  bio_type: dna
  input:
    refs: "${WORKSPACES_DIR}/_shared/inputs/refs.csv"
    name_col: ref_name
    seq_col: sequence
  permute:
    protocol: scan_dna
    params:
      regions: []
  output:
    dir: "${WORKSPACE_DIR}/outputs"
    layout: flat
```

Supported config path tokens are `${WORKSPACE_DIR}`, `${WORKSPACES_DIR}`,
`${PERMUTER_RESOURCE_DIR}`, environment variables, and `~`. `${JOB_DIR}` is a
contract error.

## CLI

```bash
uv run permuter --help
uv run permuter workspace list --root src/dnadesign/permuter/workspaces
uv run permuter workspace validate --workspace src/dnadesign/permuter/workspaces/rnaseh1_nt_scan
uv run permuter run --workspace rnaseh1_nt_scan --ref BL21_RNase_H1_wt
uv run permuter evaluate --workspace rnaseh1_nt_scan --ref BL21_RNase_H1_wt --with smoke:placeholder:log_likelihood
uv run permuter plot --workspace rnaseh1_nt_scan --ref BL21_RNase_H1_wt --metric-id smoke
uv run permuter validate --data src/dnadesign/permuter/workspaces/rnaseh1_nt_scan/outputs/records.parquet --strict
```

`--workspace` accepts a workspace directory, a `config.yaml` path, or a scope id
searched under `$PERMUTER_WORKSPACES`, `./workspaces`, and the packaged
Permuter workspaces. `--out` is an explicit output override. If
`PERMUTER_OUTPUT_ROOT` is set, it behaves as a federated output root and writes
to `$PERMUTER_OUTPUT_ROOT/<scope>`; `--out` still wins.
`evaluate` and `plot` resolve exactly the configured workspace/ref dataset path
instead of probing alternate legacy layouts.
For `evaluate`, explicit `--with` or `--metric` entries replace workspace
`evaluate.metrics`; workspace metrics are used only when no explicit metric is
provided.

## Dataset Shape

For `layout: flat`, outputs are:

```text
workspaces/<scope>/outputs/
  records.parquet
  REF.fa
  REF_AA.fa                  # when the refs table provides an AA column
  RECORD.md
  plots/
```

The Parquet table contains USR core columns plus canonical Permuter columns such
as `permuter__scope`, `permuter__var_id`, `permuter__ref`, `permuter__protocol`,
`permuter__modifications`, and `permuter__observed__<metric_id>`.

## Public API

Sibling tools and studies import from `dnadesign.permuter`, never
`dnadesign.permuter.src.*`:

```python
from dnadesign.permuter import (
    CodingDnaDmsRequest,
    NucleotideDmsRequest,
    default_codon_table_path,
    generate_variants,
)

dna_result = generate_variants(
    NucleotideDmsRequest(
        ref_name="toy",
        sequence="ACGT",
        metadata={"study": "example"},
    )
)

rt_result = generate_variants(
    CodingDnaDmsRequest(
        ref_name="rt_cds",
        sequence="AAA",
        codon_table=default_codon_table_path("ecoli"),
        positions=(1,),
        max_variants=500,
        metadata={"study": "example", "slot_id": "rt_cds"},
    )
)
```

The public API is filesystem-free and returns typed in-memory records for
nucleotide, protein, and coding-DNA-backed DMS requests. Coding-DNA requests can
set `max_variants` to fail before materializing oversized scans.
