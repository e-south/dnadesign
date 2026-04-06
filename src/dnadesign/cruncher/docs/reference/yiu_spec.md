## YIU Spec Reference

**Audience:** YIU workflow users and maintainers
**Applies to:** `configs/yiu/*.yiu.yaml`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-05
**Last updated by:** cruncher-maintainers on 2026-04-05

YIU uses one strict v4 payload-rendering document root. A YIU spec tells Cruncher where the payload comes from, which 4 nt junction windows and mismatch plans are allowed, how PWM context should be resolved, and where the published bundle should be written.

Use [YIU Workflow](../guides/yiu_workflow.md) for command flow, [YIU Artifacts](yiu_artifacts.md) for emitted files and `show`, and [YIU Visual System](yiu_visual_system.md) for view hierarchy.

<!-- docs:toc:off -->

### Quick links

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](yiu_artifacts.md)
- [YIU Visual System](yiu_visual_system.md)
- [Sampling and Analysis](../guides/sampling_and_analysis.md)

### Input-side workspace adjacency

These are the input-side files that usually sit next to a YIU spec. Bundle layout, rendered PDFs, and inspection surfaces are described in [YIU Artifacts](yiu_artifacts.md).

```text
configs/
  runbook.yaml
  yiu/
    <workflow>.yiu.yaml
motifs/
  <workflow>_pwm_context.yaml   # optional
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
    mode: center_locked
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

output:
  bundle_dir: outputs/plots/yiu__tetr_monotypic_hit
  emit_render_jobs_debug: false
```

### Contract rules

- `yiu.contract` must equal `split_yiu_payload_rendering_v4`
- `schema_version` must equal `1`
- `input.kind` must be `user_sequence` or `sample_hit`
- exactly one matching input block must be populated
- `optimization.junction.overhang_length` must equal `4`
- `optimization.mismatches.count` must be `1` or `2`
- `optimization.mismatches.strand_mode` must be `per_position`
- YIU v4 payload inputs must contain exact `A/C/G/T` bases; ambiguous IUPAC symbols are rejected
- `output.bundle_dir` is workspace-relative and required
- `output.published_plot_path` is workspace-relative when present and must point to a `.pdf`
- YIU always publishes three fixed view contracts; there is no opt-out flag
- YIU does not accept legacy state-graph fields, owners, enzymes, or external-part directives
- YIU v4 is mismatch-centric, not bulge-aware; legacy `bulge_mask` and `split` keys are rejected

### Input normalization

`user_sequence` must provide one exact payload sequence using `A/C/G/T` only.

`sample_hit` must provide:

- `hit_id`
- `sample_name`

Optional provenance fields:

- `payload_sequence`
- `source_artifact_path`
- `source_artifact`
- `metadata`

YIU may adapt a public Sample artifact when the artifact yields one exact payload sequence with `A/C/G/T` only. Supported public hit-table shapes currently include:

- `export/table__elites.csv` with `elite_id` / `elite_sequence`
- YIU-owned hit tables with `hit_id` / `payload_sequence`
- `outputs/optimize/tables/elites.parquet` with `id` / `sequence`

YIU accepts three stable payload-source shapes for `sample_hit`:

- direct `payload_sequence`
- workspace-local `source_artifact_path`
- sibling-workspace public artifact references through `metadata.source_workspace` + `metadata.source_artifact`

Relative `source_artifact_path` values are resolved inside the current workspace only. `metadata.source_workspace` is explicit: use an absolute path or a sibling workspace path or name that resolves from the current workspace root or its parent directory. Ambiguous or missing sources fail fast.

The most common `sample_hit` handoff is a Sample public hit table such as `outputs/optimize/tables/elites.parquet`.

### Maintainer seams

- `yiu/spec_models.py` is the stable public schema facade; focused input, PWM, and rendering validators live in `yiu/spec_input_models.py`, `yiu/spec_pwm_models.py`, and `yiu/spec_rendering_models.py`.
- `yiu/payload_resolution.py` is the stable public input-resolution seam; sample-hit artifact lookup and table loading live in `yiu/sample_hit_sources.py`.
- `yiu/pwm_context.py` is the stable public PWM-resolution seam; inline/file dispatch lives in `yiu/pwm_context_sources.py`, sample-context orchestration lives in `yiu/pwm_context_sample_context.py`, occurrence-table loading lives in `yiu/pwm_context_sample_occurrences.py`, and motif-instance materialization lives in `yiu/pwm_context_sample_motifs.py`.
- Keep schema and source-resolution changes inside those focused helpers unless the public facade contract itself is changing.

### Junction and PWM rules

- `optimization.junction.max_payload_body_length` is part of junction validity for all modes: the selected window must leave left and right payload bodies less than or equal to that bound
- `junction.mode: center_locked` searches valid internal 4 nt windows and chooses the candidate closest to the payload midpoint
- `junction.mode: derived` is accepted as a legacy alias for `center_locked`, and normalized outputs emit `center_locked`
- `junction.mode: explicit_window` is allowed only when the window is internal and also satisfies `max_payload_body_length`
- `junction.mode: optimize` enumerates all valid windows and mismatch plans exhaustively, ranking candidates by PWM or log-likelihood retention first and midpoint proximity second
- `optimization.pwm.objective.secondary` uses the fixed ladder `total_loss`, `midpoint_proximity`, `terminal_position_avoidance`, `default_strand_preference`, and `lexical_stability`
- legacy specs that still list `body_length_balance` are accepted and normalized to that fixed ladder because the metric is redundant with `midpoint_proximity` for a fixed 4 nt junction
- the normalized model stores `junction.start`, `junction.end`, `selected_payload_sequence`, and `selected_complement_sequence`
- `mismatches.candidate_positions` are zero-based offsets inside the 4 nt junction window `[0, 1, 2, 3]`
- terminal positions `0` and `3` are opt-in only
- PWM mode `none` disables scoring even if a source is available
- PWM mode `use_if_available` records a deterministic fallback reason when context is unavailable
- PWM mode `require` fails fast when context is missing, malformed, or ambiguous
- `optimization.pwm.source.kind: sample_context` is only valid with `input.kind: sample_hit`

### Derived normalized payload fields

Every valid spec derives:

- payload forward, aligned-complement, and reverse-complement sequences
- one internal `junction` window with exact left body, right body, payload-forward sequence, and selected complement sequence
- a normalized mismatch plan with one mutated strand per mismatch position
- PWM context metadata when available
- top-level optimization decision fields used by `validate`, `render`, and `show`

The derived split-row publication exposes row-2 display truth separately from the normalized payload object. The payload view uses `yiu_payload_visual_v1` so PWM motif layers can be added without changing the split or assembled view contracts.

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contracts.
