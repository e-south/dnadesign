## YIU Spec Reference

**Audience:** YIU workflow users and maintainers
**Applies to:** `configs/yiu/*.yiu.yaml`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-08
**Last updated by:** cruncher-maintainers on 2026-04-08

YIU uses one strict v4 payload-rendering document root. A YIU spec tells Cruncher where the payload comes from, which 4 nt junction windows and mismatch plans are allowed, how PWM context should be resolved, and where the published bundle should be written.

Use this page when you are authoring or reviewing `.yiu.yaml` fields. Use the workflow guide for command flow and solver behavior, the artifacts page for bundle contents and `show`, and the visual-system page for render hierarchy.

Use [YIU Workflow](../guides/yiu_workflow.md) for command flow, [YIU Artifacts](yiu_artifacts.md) for emitted files and `show`, and [YIU Visual System](yiu_visual_system.md) for view hierarchy.

<!-- docs:toc:off -->

### Quick links

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](yiu_artifacts.md)
- [YIU Visual System](yiu_visual_system.md)
- [Sampling and Analysis](../guides/sampling_and_analysis.md)

### Files next to a spec

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
    candidate_positions: [0, 1, 2, 3]
    allowed_strands: [complement, payload]
    strand_mode: per_position
    default_strand_preference: complement
    ligation_profile: t4
    ligation_awareness_mode: secondary
    ligation_selection_mode: secondary
    bad_pattern_heuristics: false
  pwm:
    mode: none
    source:
      kind: none
    objective:
      primary: maximin
      secondary:
        - total_loss
        - ligation_awareness
        - midpoint_proximity
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
    source_artifact_path: outputs/optimize/tables/elites.parquet
    metadata:
      tf_name: tetR
      source_workspace: demo_monotypic_tetr
      payload_label: tetr_payload

output:
  bundle_dir: outputs/plots/yiu__tetr_monotypic_hit
  emit_render_jobs_debug: false
```

### How the solver uses this spec

- `input` resolves one exact payload sequence from either `user_sequence` or `sample_hit`.
- `optimization.junction` limits which internal 4 nt windows are legal.
- `optimization.mismatches` controls exhaustive enumeration across offsets, mutated strands, and mismatch count.
- `optimization.mismatches.ligation_selection_mode` decides whether ligation stays advisory, PWM-tolerance-gated, or hard-filtered before ranking survivors.
- `optimization.pwm` controls the primary scoring surface and whether degraded PWM mode is visible or fatal.
- `output` controls where the deterministic bundle is written; it does not change the biological search.

### Contract rules

- `yiu.contract` must equal `split_yiu_payload_rendering_v4`
- `schema_version` must equal `1`
- `input.kind` must be `user_sequence` or `sample_hit`
- exactly one matching input block must be populated
- `optimization.junction.overhang_length` must equal `4`
- `optimization.mismatches.count` must be `1` or `2`
- `optimization.mismatches.strand_mode` must be `per_position`
- `optimization.mismatches.ligation_profile` must be one of `none`, `t4`, `t7`, `t3`, `pbcv1`, or `hlig3`
- `optimization.mismatches.ligation_awareness_mode` must be `disabled` or `secondary`
- `optimization.mismatches.ligation_selection_mode` must be `secondary`, `pwm_tolerance_then_ligation`, or `hard_ligation_filter`
- `optimization.mismatches.bad_pattern_heuristics` is optional and only valid when ligation awareness is active
- `optimization.mismatches.pwm_worst_loss_tolerance` and `optimization.mismatches.pwm_total_loss_tolerance` are only valid with `ligation_selection_mode: pwm_tolerance_then_ligation`
- `optimization.mismatches.max_worst_mismatch_class_tier`, `max_middle_mismatch_count`, `allow_double_middle`, and `allow_tnna_like_overhangs` are only valid with `ligation_selection_mode: hard_ligation_filter`
- if `optimization.mismatches.candidate_positions` is omitted, YIU defaults to `[0, 1, 2, 3]`
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
- `metadata`

YIU may adapt a public Sample artifact when the artifact yields one exact payload sequence with `A/C/G/T` only. Supported public hit-table shapes currently include:

- `export/table__elites.csv` with `elite_id` / `elite_sequence`
- YIU-owned hit tables with `hit_id` / `payload_sequence`
- `outputs/optimize/tables/elites.parquet` with `id` / `sequence`

YIU accepts three stable payload-source shapes for `sample_hit`:

- direct `payload_sequence`
- workspace-local `source_artifact_path`
- sibling-workspace public artifact references through `metadata.source_workspace` + `source_artifact_path`

Relative `source_artifact_path` values are resolved inside the current workspace by default. When `metadata.source_workspace` is provided, the same relative `source_artifact_path` is resolved inside that explicit sibling workspace instead. Ambiguous or missing sources fail fast.

The most common `sample_hit` handoff is a Sample public hit table such as `outputs/optimize/tables/elites.parquet`.

### Maintainer seams

- `yiu/spec_models.py` is the stable public schema facade; focused input, PWM, and rendering validators live in `yiu/spec_input_models.py`, `yiu/spec_pwm_models.py`, and `yiu/spec_rendering_models.py`.
- `yiu/payload_resolution.py` is the stable public input-resolution seam; sample-hit artifact lookup and table loading live in `yiu/sample_hit_sources.py`.
- `yiu/pwm_context.py` is the stable public PWM-resolution seam; inline/file dispatch lives in `yiu/pwm_context_sources.py`, sample-context orchestration lives in `yiu/pwm_context_sample_context.py`, occurrence-table loading lives in `yiu/pwm_context_sample_occurrences.py`, and motif-instance materialization lives in `yiu/pwm_context_sample_motifs.py`.
- Keep schema and source-resolution changes inside those focused helpers unless the public facade contract itself is changing.

### Junction and PWM rules

- `optimization.junction.max_payload_body_length` is part of junction validity for all modes: the selected window must leave left and right payload bodies less than or equal to that bound
- `junction.mode: center_locked` searches valid internal 4 nt windows and chooses the candidate closest to the payload midpoint
- `junction.mode: explicit_window` is allowed only when the window is internal and also satisfies `max_payload_body_length`
- `junction.mode: optimize` enumerates all valid windows and mismatch plans exhaustively, then ranks candidates by the active ligation policy plus PWM or log-likelihood retention before midpoint proximity and deterministic tie-breaks settle any remaining ties
- `optimization.pwm.objective.secondary` uses the fixed ladder `total_loss`, `ligation_awareness`, `midpoint_proximity`, `default_strand_preference`, and `lexical_stability`
- the normalized model stores `junction.start`, `junction.end`, `selected_payload_sequence`, and `selected_complement_sequence`
- `mismatches.candidate_positions` are zero-based offsets inside the 4 nt junction window `[0, 1, 2, 3]`
- use `[0, 1, 2, 3]` when you want ligation-aware ranking to compare edge (`0`, `3`) versus middle (`1`, `2`) positions
- limiting the pool to `[1, 2]` intentionally keeps the search middle-only and makes edge-vs-middle comparison unavailable in the winning plan
- `ligation_profile=none` preserves legacy ranking and should be read as legacy mode, not as a quietly disabled secondary mode
- `ligation_awareness_mode=disabled` makes ligation-aware scoring inert even when a ligation profile is present
- `ligation_selection_mode=secondary` preserves the current PWM-first ladder when ligation is active
- `ligation_selection_mode=pwm_tolerance_then_ligation` keeps only candidates inside the declared PWM loss budget, then lets ligation outrank small PWM differences inside that admissible set
- `ligation_selection_mode=hard_ligation_filter` drops candidates that violate the configured ligation limits before PWM and geometry tie-breaks are allowed to act
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

## Ligation-aware mismatch ranking

YIU can optionally apply ligation-aware ranking for 4-bp junction mismatches. This ranking is based on Bilotti et al. (Nucleic Acids Research, 2022), who profiled mismatch discrimination during end-joining by several DNA ligases. In those data, G:T/T:G mismatches were the most commonly tolerated across ligases, mismatches near the ligation seam were better tolerated than mismatches in the middle of the 4-bp overhang, and T3/PBCV-1/hLig3 were more permissive of G:A/G:G than T4/T7. YIU uses these observations as deterministic ranking heuristics after PWM-preservation scoring, not as hard physical guarantees for any specific construct.

YIU stores junction offsets in payload-forward coordinates `0..3` and scores on aligned duplex coordinates. Human-facing payload, split, and assembled views may rewrite strands into explicit 5' to 3' display. Ligation-aware scoring therefore derives mismatch class from the final duplex base pair and does not depend on whether the payload or complement strand was mutated.

`ligation_profile=none` preserves legacy behavior.
That legacy behavior should be read as legacy mode, not as a quietly disabled secondary mode.
`ligation_profile=t4` is the recommended default for T4-like assembly workflows.
When ligation-aware scoring is active but the selected winning plan only uses middle positions, bundle summaries and CLI output mark edge-vs-middle comparison as unavailable for that plan.
`bad_pattern_heuristics` is the TNNA-style penalty heuristic only; it is not a general mismatch-quality classifier.

YIU exposes three explicit ligation selection modes:

- `secondary`: the current behavior. PWM stays primary when effective, then ligation-aware ranking acts as the secondary ladder.
- `pwm_tolerance_then_ligation`: YIU finds the best PWM losses first, keeps candidates within `pwm_worst_loss_tolerance` and `pwm_total_loss_tolerance`, and then lets ligation outrank small PWM differences inside that admissible set.
- `hard_ligation_filter`: YIU drops candidates that violate the explicit ligation contract before ranking survivors. The strict defaults are `max_worst_mismatch_class_tier: 0`, `max_middle_mismatch_count: 1`, `allow_double_middle: false`, and `allow_tnna_like_overhangs: false`.

The legacy alias `hard_filter` is still accepted in specs, but YIU normalizes and reports that mode as `hard_ligation_filter`.

When ligation awareness is active and PWM is effective, `secondary` keeps PWM loss primary and then ranks by mismatch-class tier, middle-mismatch count, double-middle penalty, and optional bad-pattern heuristics before falling back to midpoint distance, strand preference, and lexical stability. `pwm_tolerance_then_ligation` moves ligation earlier, but only inside the declared PWM loss budget. `hard_ligation_filter` never silently degrades: if it removes every candidate, YIU fails fast and nudges the user toward the shortest relevant relaxation list, such as `max_worst_mismatch_class_tier`, `max_middle_mismatch_count`, `allow_double_middle`, or `allow_tnna_like_overhangs`, using the remaining candidate pool as the guide. When PWM is absent or ineffective, the ligation key becomes the leading comparator. Candidate generation stays exhaustive; the biology rules only affect ranking or admissibility. When `candidate_positions` is omitted, that exhaustive search covers all payload-forward offsets `0..3`.

### How to read candidate counts

YIU reports two useful ligation-policy counts:

- `candidate_count_before_filter`: the full enumerated pool before ligation policy is applied
- `candidate_count_after_filter`: the surviving pool after ligation policy is applied

Those counts come from exhaustive generation, not sampling. With the common full-pool settings `candidate_positions: [0, 1, 2, 3]` and `allowed_strands: [complement, payload]`:

- `count: 1` gives `4 × 2 × 3 = 24` candidates per feasible junction window
- `count: 2` gives `C(4,2) × 2^2 × 3^2 = 216` candidates per feasible junction window

The multiplication is literal: choose window, choose mismatch offsets, choose mutated strand at each offset, then choose the non-native base at each mutated site. Each site contributes `3` base choices because YIU excludes the native base from the mutation set. Use [YIU Workflow](../guides/yiu_workflow.md) for the worked derivation and examples.

### Why strict mode filters so aggressively

Under the strict `hard_ligation_filter` defaults, YIU keeps only candidates that satisfy all of these:

- worst mismatch tier no worse than `GT`-class tier `0`
- no more than one middle-position mismatch
- no `(1,2)` double-middle geometry
- no `TNNA`-like final overhang

The first rule dominates. At a single chosen site there are `6` local mutation choices: `2` strand choices times `3` non-native bases. In the standard Watson-Crick background, only one of those six yields canonical `GT`, so `hard_ligation_filter` often removes `97%` to `98%` of a two-mismatch pool. PWM still matters after that, but only for ranking the survivors. It does not inflate or shrink the `before` or `after` counts in `hard_ligation_filter`.

The paper does not isolate every exact two-mismatch geometry that YIU can generate. The strongest direct support is for G:T dominance, edge better than middle, T4/T7 versus T3/PBCV-1/hLig3 permissiveness differences, and TNNA inefficiency. Penalties such as `double_middle_flag` are engineering extrapolations grounded in the paper, not direct one-to-one measurements for every possible YIU candidate geometry.

The derived split-row publication exposes row-2 display truth separately from the normalized payload object. The payload view uses `yiu_payload_visual_v1` so PWM motif layers can be added without changing the split or assembled view contracts.

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contracts.
