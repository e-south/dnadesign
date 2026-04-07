## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-07
**Last updated by:** cruncher-maintainers on 2026-04-07

YIU turns one payload sequence into a checked junction-mismatch bundle. It accepts either an exact `user_sequence` or a `sample_hit` resolved from public Cruncher Sample outputs, searches valid 4 nt internal junction plans plus one or two mismatches, optionally scores those candidates against PWM context, and publishes three BaseRender-ready views.

<!-- docs:toc:off -->

Useful links:

- [YIU Workspace Demo](../demos/demo_yiu_workspace.md)
- [Sampling and Analysis](../guides/sampling_and_analysis.md)
- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
- [YIU Visual System](../reference/yiu_visual_system.md)
- [Cruncher architecture](../reference/architecture.md)

The public lane is:

`input payload -> normalized payload -> optimized junction/mismatch plan -> published bundle -> BaseRender`

### Pick the shortest path

- Use [YIU Workspace Demo](../demos/demo_yiu_workspace.md) for the checked-in user-sequence workspace and runbook.
- Use `cruncher yiu validate` when you only need schema, source-resolution, and payload-plan checks.
- Use `cruncher yiu render` when you need the published bundle. Add `--emit-renders` when you also want the composite PDF.
- Use `cruncher yiu show` when you need one fail-fast inspection surface for manifest, inventory, payload summary, and rendered artifacts.
- Use [YIU Spec Reference](../reference/yiu_spec.md) for input rules, [YIU Artifacts](../reference/yiu_artifacts.md) for emitted files and `show`, and [YIU Visual System](../reference/yiu_visual_system.md) for view hierarchy.

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

YIU is mismatch-centric. The junction is always a 4 nt internal window. The optimizer chooses the window, mismatch positions, mutated strands, and mutated bases within the rules in the spec. Legacy bulge and topology keys are rejected rather than guessed.

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
- `optimize`: search valid internal windows around the midpoint and rank candidates by PWM/log-likelihood retention first, ligation awareness second when enabled, then midpoint proximity and the remaining deterministic tie-break ladder.

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

### What `render` writes

`cruncher yiu render` validates the spec, reruns normalization and optimization, and writes one deterministic bundle under `output.bundle_dir`, usually `outputs/<workflow>/`.

With `--emit-renders`, YIU also renders one composite `payload_views.pdf` page and mirrors that PDF to `output.published_plot_path` when configured.

Use [YIU Artifacts](../reference/yiu_artifacts.md) for the exact emitted files. For most operators, the handoff surface is:

- `bundle_summary.json`
- `payload_views.pdf`
- `cruncher yiu show --bundle <bundle_dir>`

That summary artifact now makes the reference duplex and mismatch-present duplex explicit side by side, with payload 5' to 3' and complement 3' to 5' spelled out directly.
The current operator surface goes one step further: `bundle_summary.json` and default `cruncher yiu show` publish one `views` block for `payload`, `split_left`, `split_right`, and `assembled`, with reference and mismatch-present top/bottom rows all rewritten into explicit 5' to 3' orientation plus `changed_rows`.

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

Default human-readable `show` keeps the payload handoff in the foreground: one ligation summary line, one overhang summary, then payload, split-left, split-right, and assembled views with reference-vs-mismatch-present top/bottom rows in 5' to 3', followed by compact mismatch edits and PWM state. Default `show --json` stays operator-focused and omits the machine ledger paths plus normalized payload detail unless `--verbose` is set. Human-readable `--verbose` adds provenance, bundle contract, render/integrity details, machine-facing artifact paths, and split-row debug lines; the optimizer trace and motif context remain JSON-only.

The split middle row renders `split_payload_left` before `split_payload_right`. Each panel shows the retained fragment, its inward-facing sticky end, selected-versus-reference sticky-end metadata, the fragment-display payload-body slice, and optional ghosted excision context. The bundle summary now also publishes those retained left and right fragment duplexes as explicit top/bottom 5' to 3' rows for both reference and mismatch-present variants.

The assembled payload returns to original payload order. It publishes one explicit `junction_span` in payload coordinates rather than a seam surrogate.

### Ligation-aware mismatch ranking

YIU can optionally apply ligation-aware ranking for 4-bp junction mismatches. This ranking is based on Bilotti et al. (Nucleic Acids Research, 2022), who profiled mismatch discrimination during end-joining by several DNA ligases. In those data, G:T/T:G mismatches were the most commonly tolerated across ligases, mismatches near the ligation seam were better tolerated than mismatches in the middle of the 4-bp overhang, and T3/PBCV-1/hLig3 were more permissive of G:A/G:G than T4/T7. YIU uses these observations as deterministic ranking heuristics after PWM-preservation scoring, not as hard physical guarantees for any specific construct.

YIU stores junction offsets in payload-forward coordinates `0..3` and scores on aligned duplex coordinates. Human-facing payload, split, and assembled views may rewrite strands into explicit 5' to 3' display. Ligation-aware scoring therefore derives mismatch class from the final duplex base pair and does not depend on whether the payload or complement strand was mutated.

`ligation_profile=none` preserves legacy behavior. `ligation_profile=t4` is the recommended default for T4-like assembly workflows.

Candidate generation does not change when ligation awareness is enabled. YIU still enumerates all feasible windows, mismatch offsets, strand assignments, and non-native bases; the Bilotti-derived rules only change the deterministic ranking layer.

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
