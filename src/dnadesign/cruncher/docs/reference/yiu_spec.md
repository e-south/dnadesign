## YIU Spec Reference

**Audience:** YIU workflow users and maintainers
**Applies to:** `configs/yiu/*.yiu.yaml`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-04

YIU ships one strict v4 payload-rendering document root.
This page owns schema and normalization only. Bundle layout, render-status semantics, and operator inspection fields live in [YIU Artifacts](yiu_artifacts.md).
Use [YIU Workflow](../guides/yiu_workflow.md) for operator flow and visual posture instead of repeating bundle/output narratives here.

### Scope

- This page owns the input contract, normalization rules, and optimization rules.
- This page does not own emitted bundle layout, PDF mirroring, or bundle-drift checks.
- When you need operator flow, use the workflow guide instead of expanding the schema page.

### Recommended workspace layout

```text
configs/
  runbook.yaml
  yiu/
    <workflow>.yiu.yaml
motifs/
  <workflow>_pwm_context.yaml   # optional
outputs/
  <workflow>/
  <workflow>__payload_views.pdf
```

### Root contract

```yaml
yiu:
  schema_version: 1
  contract: split_yiu_payload_rendering_v4
  name: <workflow>

input:
  kind: user_sequence
  user_sequence:
    sequence: ACGTACGTACGT

optimization:
  junction:
    mode: optimize
    overhang_length: 4
    max_payload_body_length: 12
  mismatches:
    count: 1
    candidate_positions: [1, 2]
    allowed_strands: [complement, payload]
    strand_mode: per_position
    default_strand_preference: complement
  pwm:
    mode: none
    source:
      kind: none
    objective:
      primary: maximin
      secondary:
        - total_loss
        - midpoint_proximity
        - body_length_balance
        - terminal_position_avoidance
        - default_strand_preference
        - lexical_stability

output:
  bundle_dir: outputs/<workflow>
  published_plot_path: outputs/<workflow>__payload_views.pdf
  emit_render_jobs_debug: false
```

The alternate first-class input is `sample_hit`:

```yaml
input:
  kind: sample_hit
  sample_hit:
    hit_id: tetr-monotypic-001
    sample_name: tetr
    payload_sequence: CTCTATATCTGATATAGAG
    metadata:
      tf_name: tetR
      source_workspace: demo_monotypic_tetr
      source_artifact: outputs/optimize/tables/elites.parquet
      payload_label: tetr_payload
```

### Contract rules

- `yiu.contract` must equal `split_yiu_payload_rendering_v4`
- `schema_version` must equal `1`
- `input.kind` must be `user_sequence` or `sample_hit`
- exactly one matching input block must be populated
- `optimization.junction.overhang_length` must equal `4`
- `optimization.mismatches.count` must be `1` or `2`
- `optimization.mismatches.strand_mode` must be `per_position`
- `output.bundle_dir` is workspace-relative and required
- `output.published_plot_path` is workspace-relative when present and must point to a `.pdf`
- YIU always publishes three canonical view contracts; there is no opt-out flag
- YIU does not accept legacy state-graph fields, owners, enzymes, or external-part directives

### Input normalization

`user_sequence` must provide one exact payload sequence.

`sample_hit` must provide:

- `hit_id`
- `sample_name`

Optional provenance fields:

- `payload_sequence`
- `source_artifact_path`
- `source_artifact`
- `metadata`

YIU may adapt a public Sample artifact when the artifact yields one exact payload sequence. Supported public hit-table shapes currently include:

- `export/table__elites.csv` with `elite_id` / `elite_sequence`
- YIU-owned hit tables with `hit_id` / `payload_sequence`
- `outputs/optimize/tables/elites.parquet` with `id` / `sequence`

YIU accepts three stable payload-source shapes for `sample_hit`:

- direct `payload_sequence`
- workspace-local `source_artifact_path`
- sibling-workspace public artifact references through `metadata.source_workspace` + `metadata.source_artifact`

Relative `source_artifact_path` values are resolved inside the current workspace only. `metadata.source_workspace` is explicit: use an absolute path or a sibling workspace path/name that resolves from the current workspace root or its parent directory. Ambiguous or missing sources fail fast.

### Junction and PWM rules

- `junction.mode: derived` searches valid internal 4 nt windows and chooses the candidate closest to the payload midpoint
- `junction.mode: explicit_window` is allowed only when the window is internal and leaves non-empty left and right payload bodies
- `junction.mode: optimize` enumerates all valid windows and mismatch plans exhaustively
- the normalized model stores `junction.start`, `junction.end`, `selected_payload_sequence`, and `selected_complement_sequence`
- `mismatches.candidate_positions` are zero-based offsets inside the 4 nt junction window `[0, 1, 2, 3]`
- terminal positions `0` and `3` are opt-in only
- PWM mode `none` disables scoring even if a source is available
- PWM mode `use_if_available` records a deterministic fallback reason when context is unavailable
- PWM mode `require` fails fast when context is missing, malformed, or ambiguous

### Published metadata

Every valid spec derives:

- canonical payload forward/aligned-complement/reverse-complement sequences
- one internal `junction` window with exact left body, right body, payload-forward sequence, and selected complement sequence
- a normalized mismatch plan with one mutated strand per mismatch position
- PWM context metadata when available
- top-level optimization decision fields used by `validate`, `render`, and `show`

The derived split-row publication exposes row-2 display truth separately from the canonical normalized payload object. The payload view uses `yiu_payload_visual_v1` so PWM motif layers can be added without changing the split/assembled view contracts.

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contracts.
