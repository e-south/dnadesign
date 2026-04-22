## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-08
**Last updated by:** cruncher-maintainers on 2026-04-08

YIU turns one payload sequence into a checked junction-mismatch bundle. It accepts either an exact `user_sequence` or a `sample_hit` resolved from public Cruncher Sample outputs, searches valid 4 nt internal junction plans plus one or two mismatches, optionally scores those candidates against PWM context, and publishes three BaseRender-ready views.

Use this guide when you need command flow and solver behavior. Use the spec reference for field-by-field schema, the artifacts page for bundle contents and `show`, and the visual-system page for render hierarchy.

<!-- docs:toc:off -->

Use other pages for:

- [YIU Workspace Demo](../demos/demo_yiu_workspace.md)
- [Sampling and Analysis](../guides/sampling_and_analysis.md)
- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
- [YIU Visual System](../reference/yiu_visual_system.md)
- [Cruncher architecture](../reference/architecture.md)

The public lane is:

`input payload -> normalized payload -> optimized junction/mismatch plan -> published bundle -> BaseRender`

### YIU docs route

1. Start with [YIU Workspace Demo](../demos/demo_yiu_workspace.md) for the checked-in user-sequence workspace and runbook.
2. Stay in this guide for the public `init-workspace -> validate -> render -> show` flow and the ranking logic.
3. Move to [YIU Spec Reference](../reference/yiu_spec.md) when you are editing `.yiu.yaml` fields, defaults, or degraded-mode behavior.
4. Move to [YIU Artifacts](../reference/yiu_artifacts.md) when you need emitted files, `show`, or bundle-integrity rules.
5. Move to [YIU Visual System](../reference/yiu_visual_system.md) when you need payload/split/assembled hierarchy or render emphasis.

### Inputs and published views

The checked-in reference workspace lives at `src/dnadesign/cruncher/workspaces/demo_yiu_payload`.

YIU accepts two first-class inputs:

- `user_sequence`
- `sample_hit`

Both inputs normalize into one payload object and publish exactly three views:

- `payload`
- `split_payload`
- `assembled_payload`

The public contract is `split_yiu_payload_rendering_v4`.

YIU is mismatch-centric. The junction is always a 4 nt internal window. Legacy bulge and topology keys are rejected rather than guessed.

### How YIU chooses a plan

1. Resolve one exact payload from either `user_sequence` or `sample_hit`. Ambiguous or missing sources fail fast.
2. Build the valid internal 4 nt junction windows allowed by the junction mode and payload-body bounds.
3. Enumerate mismatch plans exhaustively across the allowed junction offsets, mismatch count, strand assignments, and non-native substitutions. If you omit `optimization.mismatches.candidate_positions`, YIU uses `[0, 1, 2, 3]`.
4. Rank the candidates. PWM or log-likelihood retention stays primary when effective unless `ligation_selection_mode` says otherwise. `secondary` keeps ligation advisory, `pwm_tolerance_then_ligation` gates a PWM-near set before ligation ranking, and `hard_ligation_filter` removes inadmissible ligation plans before PWM and geometry ranking act on survivors.
5. Publish one deterministic bundle with `payload`, `split_payload`, and `assembled_payload` views.

Middle-only pools such as `[1, 2]` are still allowed, but the resulting bundle will say edge-vs-middle comparison is unavailable for the winning plan.

### Where `sample_hit` comes from

YIU can reuse public Sample outputs instead of starting from a hand-written payload sequence.

`sample_hit` supports three stable source shapes:

- a direct `payload_sequence`
- a workspace-local `source_artifact_path`
- a sibling-workspace reference through `metadata.source_workspace` plus `source_artifact_path`

The common handoff is a Sample public hit table such as `outputs/optimize/tables/elites.parquet`. When `optimization.pwm.source.kind: sample_context` is selected, YIU also resolves motif context from the same Sample-backed payload source. Ambiguous or missing sources fail fast.

### Command surface

```bash
uv run cruncher yiu init-workspace WORKSPACE
uv run cruncher yiu init-workspace WORKSPACE --sequence AACCGGTTGGTT --junction-mode center_locked
uv run cruncher yiu validate --spec configs/yiu/<workflow>.yiu.yaml
uv run cruncher yiu validate --spec configs/yiu/<workflow>.yiu.yaml --json
uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml
uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml --emit-renders
uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml --json
uv run cruncher yiu show --bundle outputs/<workflow>
uv run cruncher yiu show --bundle outputs/<workflow> --json --verbose
```

`design` is not part of the public YIU surface.

### Minimal authoring example

Keep the first spec minimal and concrete:

```yaml
yiu:
  contract: split_yiu_payload_rendering_v4
  schema_version: 1
  name: example_payload

input:
  kind: user_sequence
  user_sequence:
    sequence: AAATTTCCCGGGAAATTTCCC

optimization:
  junction:
    mode: center_locked
  mismatches:
    count: 1
    candidate_positions: [0, 1, 2, 3]
    ligation_profile: t4
    ligation_awareness_mode: secondary
    ligation_selection_mode: secondary
    bad_pattern_heuristics: false
  pwm:
    mode: none
    source:
      kind: none

output:
  bundle_dir: outputs/example_payload
```

For v4, payload inputs must be exact `A/C/G/T` sequences. Ambiguous IUPAC symbols and legacy bulge or split topology keys are not part of the public v4 lane.
Use `candidate_positions: [0, 1, 2, 3]` when you want ligation-aware ranking to compare edge and middle offsets. Restricting the pool to `[1, 2]` intentionally keeps the search middle-only.

The main junction policies are:

- `center_locked`: choose the valid internal 4 nt window nearest the payload midpoint and keep the junction fixed there.
- `explicit_window`: use one explicit internal 4 nt window.
- `optimize`: search valid internal windows around the midpoint and rank candidates by the active ligation policy plus PWM/log-likelihood retention, then midpoint proximity and the remaining deterministic tie-break ladder.

### Ligation posture

- `ligation_profile=none` is legacy ranking, not a quietly disabled secondary mode.
- `ligation_awareness_mode=disabled` makes ligation-aware scoring inert even if a profile is configured.
- `ligation_selection_mode=secondary` preserves the existing PWM-first contract when ligation is active.
- `ligation_selection_mode=pwm_tolerance_then_ligation` keeps candidates within the declared PWM loss budget, then lets ligation outrank small PWM differences inside that admissible pool.
- `ligation_selection_mode=hard_ligation_filter` hard-gates the pool using the ligation contract before ranking survivors.
- candidate pools that exclude `0` and `3` are edge-blind by configuration, not by fallback
- `bad_pattern_heuristics` is the TNNA-style penalty heuristic only.
- The bundle summary and `show` output now name `legacy`, `inert`, `edge_blind`, and `active` ligation states explicitly, plus the selected ligation policy mode and before/after candidate counts when filtering happens.

### How candidate counts work

YIU runs in three stages: generate candidates, apply the ligation policy, then rank the survivors. The `before` and `after` counts are just the first two stages made visible.

- `before` is the full candidate pool after YIU has enumerated every feasible junction window, mismatch-position set, strand assignment, and non-native base choice.
- `after` is the subset that survives the active ligation policy.
- In `hard_ligation_filter`, PWM does not change either count. PWM only ranks the survivors once the strict ligation gate has finished.

For the common full-pool case:

- `candidate_positions: [0, 1, 2, 3]`
- `allowed_strands: [complement, payload]`
- `junction.mode: optimize`
- `count: 1` or `count: 2`

For each feasible internal 4 nt window, YIU builds candidates in four steps:

1. Choose a feasible internal 4 nt junction window.
2. Choose which offsets inside that 4 nt window will carry mismatches.
3. Choose which strand is mutated at each selected offset.
4. Choose the non-native base at each selected site.

Each chosen site has `3` base choices, not `4`, because YIU never keeps the native base when it is enumerating a mutation.

That gives clean per-window counts:

- `count: 1` gives `4 positions × 2 strands × 3 bases = 24` candidates per window.
- `count: 2` gives `C(4,2) × 2^2 × 3^2 = 6 × 4 × 9 = 216` candidates per window.

That is why workspace totals often factor into "feasible windows × per-window combinatorics". For example, `192` means `8` feasible windows with `24` single-mismatch candidates each, while `2376` means `11` feasible windows with `216` two-mismatch candidates each.

### What strict mode removes

Strict mode is the `hard_ligation_filter` policy with these defaults:

- `max_worst_mismatch_class_tier: 0`
- `max_middle_mismatch_count: 1`
- `allow_double_middle: false`
- `allow_tnna_like_overhangs: false`

In practice, that means:

- every mismatch must land in the top-tier `GT` class
- a two-mismatch plan may use at most one middle offset
- the `(1,2)` double-middle geometry is rejected
- a final overhang shaped like `T N N A` is rejected

The GT rule does most of the work. At one chosen site there are `6` local mutation choices: `2` strand choices times `3` non-native bases. In the standard Watson-Crick background YIU operates on, only one of those six local choices lands in that top-tier `GT` class. Most of the raw pool disappears right there, before middle-position and TNNA checks have a chance to act.

For the full two-mismatch pool, the best-case strict ceiling is still small:

- start from `216` raw candidates per window
- keep only the `GT`-compatible choices
- then remove the `(1,2)` double-middle pair

That leaves at most `5` survivors per window, or about `2.3%` of the raw two-mismatch pool. So filtering away `97%` to `98%` of candidates under strict mode is normal.

### Why PWM still changes the winner

Strict mode does not mean "pick the most edge-heavy survivor". It means "throw out the inadmissible plans first."

After that:

- the ligation filter decides which candidates are still legal
- PWM decides which legal candidate wins when PWM context is effective
- ligation and geometry break ties among those remaining legal candidates

That is why a strict-mode winner can still be `GT,GT edge,middle` instead of `GT,GT edge,edge`. Both are admissible. PWM may prefer the `edge,middle` survivor, and only then do the later tie-breaks apply.

### What `validate` checks

- the root contract and schema version match `split_yiu_payload_rendering_v4`
- exactly one input kind is populated
- the resolved payload sequence exists and contains exact `A/C/G/T` bases
- the junction policy yields at least one valid internal 4 nt window with non-empty left and right payload bodies
- the mismatch policy is internally consistent and keeps `strand_mode: per_position`
- PWM mode and PWM source are compatible with the input kind
- `sample_hit` provenance resolves to one exact payload sequence or fails fast
- PWM-aware optimization remains deterministic and exhaustive across valid windows, mismatch positions, strand assignments, and allowed non-native base substitutions
- legacy `bulge_mask` and `split` keys are rejected because they are not part of `split_yiu_payload_rendering_v4`
- optimizer traces in the normalized and summary surfaces are bounded samples, not full search ledgers

### What `render` writes

`cruncher yiu render` validates the spec, reruns normalization and optimization, and writes one deterministic bundle under `output.bundle_dir`, usually `outputs/<workflow>/`.

With `--emit-renders`, YIU also renders one composite `payload_views.pdf` page and mirrors that PDF to `output.published_plot_path` when configured.

Use [YIU Artifacts](../reference/yiu_artifacts.md) for the exact emitted files. For most operators, the handoff surface is:

- `bundle_summary.json`
- `payload_views.pdf`
- `cruncher yiu show --bundle <bundle_dir>`

`bundle_summary.json` and default `cruncher yiu show` keep the operator handoff tight. They publish one `views` block for `payload`, `split_left`, `split_right`, and `assembled`, with reference and mismatch-present top/bottom rows rewritten into explicit 5' to 3' orientation plus `changed_rows`. That makes the reference duplex and mismatch-present duplex easy to compare without opening the machine-facing ledgers.

The remaining published JSON files are machine-facing bundle ledgers or render contracts:

- `bundle_manifest.json`
- `normalized_payload.json`
- `visual_inventory.json`
- `payload_view.json`
- `split_payload_view.jsonl` (JSONL rows)
- `assembled_payload_view.json`

The payload view uses `yiu_payload_visual_v1`. When PWM context is effective, that view carries motif layers aligned to payload-forward coordinates. When PWM is absent or disabled, the same contract stays valid with an empty `motif_layers` list.

### What `show` checks and reports

`cruncher yiu show` reads the bundle, checks that the manifest, inventory, normalized payload, published view contracts, and rendered artifacts all agree, and then prints a concise summary or JSON.

`show` is fail-fast on bundle drift. Missing published view contracts, manifest and inventory disagreements, payload-view motif drift, a `rendered` bundle with a missing `payload_views.pdf`, or a configured published plot path that does not exist are treated as bundle corruption.

Default human-readable `show` keeps the payload handoff in the foreground:

- a ligation summary block
- one overhang summary
- payload, split-left, split-right, and assembled views with reference-vs-mismatch-present top/bottom rows in 5' to 3'
- compact mismatch edits, PWM state, and a bounded trace summary

Default `show --json` returns the operator summary plus bundle artifact paths, integrity detail, `motif_context`, and `optimization_decision`. Human-readable `--verbose` adds provenance, bundle contract, render/integrity details, and split-row debug lines; the full optimizer trace and motif context remain JSON-only.

The split middle row renders `split_payload_left` before `split_payload_right`. Each panel shows the retained fragment, its inward-facing sticky end, selected-versus-reference sticky-end metadata, the fragment-display payload-body slice, and optional ghosted excision context. The bundle summary now also publishes those retained left and right fragment duplexes as explicit top/bottom 5' to 3' rows for both reference and mismatch-present variants.
The bundle summary also includes an explicit ligation posture block and a trace sampling note so operators can tell when ligation is legacy, inert, edge-blind, or active without guessing from the selected candidate alone.

The assembled payload returns to original payload order. It publishes one explicit `junction_span` in payload coordinates rather than a seam surrogate.

### Ligation-aware mismatch ranking

YIU can optionally apply ligation-aware ranking for 4-bp junction mismatches. This ranking is based on Bilotti et al. (Nucleic Acids Research, 2022), who profiled mismatch discrimination during end-joining by several DNA ligases. In those data, G:T/T:G mismatches were the most commonly tolerated across ligases, mismatches near the ligation seam were better tolerated than mismatches in the middle of the 4-bp overhang, and T3/PBCV-1/hLig3 were more permissive of G:A/G:G than T4/T7. YIU uses these observations as deterministic ranking heuristics after PWM-preservation scoring, not as hard physical guarantees for any specific construct.

YIU stores junction offsets in payload-forward coordinates `0..3` and scores on aligned duplex coordinates. Human-facing payload, split, and assembled views may rewrite strands into explicit 5' to 3' display. Ligation-aware scoring therefore derives mismatch class from the final duplex base pair and does not depend on whether the payload or complement strand was mutated.

`ligation_profile=none` preserves legacy behavior. `ligation_profile=t4` is the recommended default for T4-like assembly workflows.

Candidate generation does not change when ligation awareness is enabled. YIU still enumerates all feasible windows, mismatch offsets, strand assignments, and non-native bases; the Bilotti-derived rules only change the deterministic ranking layer.

Three ligation policy modes are public:

- `secondary`: keep the current PWM-first ladder and use ligation as the secondary comparator.
- `pwm_tolerance_then_ligation`: admit candidates within `pwm_worst_loss_tolerance` and `pwm_total_loss_tolerance`, then let ligation outrank small PWM differences inside that set.
- `hard_ligation_filter`: reject candidates that exceed `max_worst_mismatch_class_tier`, `max_middle_mismatch_count`, `allow_double_middle`, or `allow_tnna_like_overhangs` before ranking survivors.

The strict `hard_ligation_filter` defaults are intentionally conservative and never silently degrade. If they remove the entire pool, YIU fails fast and prints a short relaxation hint naming the smallest relevant config fields to change, based on the rejected candidate pool. The legacy alias `hard_filter` is accepted for compatibility, but YIU emits and reports the normalized name `hard_ligation_filter`.

The paper does not isolate every exact two-mismatch geometry that YIU can generate. The strongest direct support is for G:T dominance, edge better than middle, T4/T7 versus T3/PBCV-1/hLig3 permissiveness differences, and TNNA inefficiency. Penalties such as `double_middle_flag` are engineering extrapolations grounded in the paper, not direct one-to-one measurements for every possible YIU candidate geometry.

### Visual direction

The current YIU visual system is `bench_strip`:

- `payload` uses `evidence_ribbon`
- `split_payload` uses `operator_strip`
- `assembled_payload` uses `operator_strip`

Use [YIU Visual System](../reference/yiu_visual_system.md) for the rationale and style-boundary rules.

### Maintainer boundaries

At the tool boundary, YIU publishes contracts and jobs; `baserender` consumes those contracts through its public API. Cross-tool integrations should not import `dnadesign.baserender.src.*`.

Keep schema and source-resolution edits narrow:

- `yiu/spec_models.py` stays the public schema facade
- `yiu/payload_resolution.py` stays the public input-resolution seam
- `yiu/pwm_context.py` stays the public PWM-resolution seam
- focused validators and source loaders stay behind `yiu/spec_input_models.py`, `yiu/spec_pwm_models.py`, `yiu/spec_rendering_models.py`, `yiu/sample_hit_sources.py`, `yiu/pwm_context_sources.py`, `yiu/pwm_context_sample_context.py`, `yiu/pwm_context_sample_occurrences.py`, and `yiu/pwm_context_sample_motifs.py`

### Related docs

- [YIU Workspace Demo](../demos/demo_yiu_workspace.md)
- [Sampling and Analysis](../guides/sampling_and_analysis.md)
- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
- [YIU Visual System](../reference/yiu_visual_system.md)
- [CLI Reference](../reference/cli.md)
