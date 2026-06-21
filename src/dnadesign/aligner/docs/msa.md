# Multiple Sequence Alignment

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20

Use `dnadesign.aligner.msa` when a workflow needs a generic aligned FASTA
bundle. This package owns FASTA validation, MAFFT preflight/execution, and
aligned-bundle manifests. It does not own study-specific roster curation,
provider fetching, conservation scoring, or mask algebra.

## Public API

```python
from pathlib import Path

from dnadesign.aligner.msa import MsaBackendSpec, MsaRequest, run_msa

request = MsaRequest(
    input_fasta=Path("source.fasta"),
    output_fasta=Path("source.aligned.fasta"),
    manifest_path=Path("source.aligned.manifest.yaml"),
    target_row_id="target",
    backend=MsaBackendSpec(backend_id="mafft"),
    command_args=("--globalpair", "--maxiterate", "1000", "--reorder"),
    timeout_seconds=None,
    stderr_path=Path("source.aligned.stderr.txt"),
    run_label="profile-a",
)

result = run_msa(request)
```

## Visualization API

Use `dnadesign.aligner.msa.visualization` for generic MSA QC sidecars after an
aligned FASTA exists. The visualization API is deliberately generic: it reads
accepted aligned FASTA files, validates the target row, writes QC YAML, a
per-position CSV, SVG tracks, an HTML report, and an index manifest. It does
not fetch source sequences, choose profile rosters, score conservation masks, or
decide designability.

```python
from pathlib import Path

from dnadesign.aligner.msa.visualization import (
    MsaVisualizationRequest,
    materialize_msa_visualizations,
)

result = materialize_msa_visualizations(
    MsaVisualizationRequest(
        alignment_root=Path("alignments"),
        output_root=Path("visualizations"),
        profile_ids=("profile-a", "profile-b"),
        target_row_id="target",
        target_sequence_hash="sha256:...",
        annotation_tracks_yaml=Path("annotation-tracks.yaml"),
        exemplar_rows_yaml=Path("exemplar-rows.yaml"),
        panel_spec_yaml=Path("panel-spec.yaml"),
    )
)
```

The package is intentionally semantic rather than flat:

```text
src/dnadesign/aligner/msa/visualization/
  contracts/        # request/result models and YAML contract readers
  materialization/  # orchestration, QC calculations, manifests, CSV/HTML
  renderers/        # SVG panel renderers and label placement
```

Add new study-neutral plotting behavior to the closest semantic package. Do
not add root-level implementation modules beside `cli.py`, and do not put
study-specific row choices, biological labels, or mask semantics into
`aligner`.

Annotation tracks are optional. When supplied, they must use
`target_ungapped_position` coordinates and are validated against every rendered
target row. They are display metadata only; downstream tools still own
conservation scoring and mask decisions.

```yaml
schema_id: dnadesign.aligner.msa.visualization.annotation_tracks
schema_version: 1
coordinate_space: target_ungapped_position
tracks:
  - id: domains
    label: Domains
    color: "#5b5f97"
    features:
      - id: feature_a
        label: Feature A
        start: 10
        end: 25
        color: "#1b9e77"
        fill_opacity: 0.12
        stroke_color: "#1b9e77"
        stroke_width: 2
        text_color: "#1b9e77"
        label_position: above
```

Exemplar rows are also optional. They render row-level windows around
annotation features so a report can show a few explicitly selected records
beside the target without turning FASTA order into implicit biological
evidence.

```yaml
schema_id: dnadesign.aligner.msa.visualization.exemplar_rows
schema_version: 1
profiles:
  profile-a:
    rows:
      - record_id: target
        label: Reference
        group: target
      - record_id: homolog_01
        label: Homolog 01
        group: example
```

Panel specs are optional. They enable selected-row whole-alignment overview
SVGs and target-position plurality/gap histograms. The spec is display-only:
it can declare visual trimming policy and row limits, but it cannot change the
aligned FASTA, conservation denominator, or downstream mask decisions.

```yaml
schema_id: dnadesign.aligner.msa.visualization.panel_spec
schema_version: 1
display_columns:
  coordinate_space: target_ungapped_position
  high_gap_trim_threshold: 0.9
  trim_policy: display_only_not_scoring
overview:
  enabled: true
  max_display_rows: 8
consensus_histogram:
  enabled: true
sidecar_note: Display sidecar only; not a conservation denominator.
```

`label_position` is optional and generic. Allowed values are `auto`, `inside`,
`above`, `below`, and `hidden`. Use it when a publication-style panel needs
separate border-fill spans and motif labels without hard-coding study-specific
layout in the renderer.

Partial reports are an explicit degraded mode:

```bash
uv run python -m dnadesign.aligner.msa.visualization \
  --alignment-root alignments \
  --output-root visualizations \
  --profile-id profile-a \
  --profile-id profile-b \
  --target-row-id target \
  --annotation-tracks-yaml annotation-tracks.yaml \
  --exemplar-rows-yaml exemplar-rows.yaml \
  --panel-spec-yaml panel-spec.yaml \
  --allow-missing-profiles
```

## Dependency Contract

MAFFT is a native bioinformatics tool and is installed through Pixi, not `uv`.
Run MAFFT-backed workflows through Pixi:

```bash
pixi run mafft --version
pixi run uv run pytest src/dnadesign/aligner/tests/msa -q
```

The MAFFT wrapper fails fast when the executable is unavailable. It writes
backend stdout to a temporary FASTA, validates that aligned FASTA, and only
then atomically publishes the final output path. Failed, interrupted, or timed
out runs do not create an accepted aligned FASTA or manifest. There is no
implicit fallback backend.

## Bundle Manifest

Every run writes a YAML manifest with:

- backend id and version
- executable path
- full command
- input and output FASTA paths
- input and output SHA256 hashes
- target row id, when declared
- environment and Pixi lock hash, when discoverable
- explicit failure policy
- elapsed seconds and backend return code
- stderr sidecar path and hash
- optional run label

## Boundary

`aligner.msa` emits aligned FASTA evidence and generic QC sidecars. Downstream
tools decide how to interpret that evidence. Study packages own roster
curation, provider policy, target-sequence hash policy, conservation scoring,
and mask/designability decisions.
