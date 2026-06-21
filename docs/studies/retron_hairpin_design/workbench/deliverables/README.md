## Retron Workbench Deliverables

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20

This lane records hypothesis-specific review and handoff deliverables. It is
not a generated-output directory. Use it to answer what a study cohort should
emit, which source owns each artifact, and where a naive agent should look
before running or interpreting output generation.

### Records

- `teto_pwm_trim_rescue_v1.yaml`: deliverable plan for the nine-design bidirectional
  TetR PWM trim rescue pilot. It links the design set, compiler spec, PWM provenance,
  materialized sequence bundle, PWM trim triptych, sequence montage video,
  review manifest, and GenBank sequence handoff.

### tetO Trim Review Outputs

Open these first when reviewing the generated package:

1. `reviews/handoff/teto-pwm-trim-rescue-v1.handoff.md`
2. `reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.png`
3. `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4`
4. `reviews/review_manifest.json`

The handoff Markdown is the ergonomic entry point. It maps compact variant ids
such as `t26-w02-17` to the generated GenBank, FASTA, and feature CSV files.
Complete metadata remains in `sequence_index.tsv` and `review_manifest.json`.

### Generate

Preferred local output root:

```text
docs/studies/retron_hairpin_design/workbench/outputs/teto_pwm_trim_rescue_v1/
```

Recommended sequence-bundle root under that directory:

```text
materialized/
```

Generate the review package after materializing the nine-design compiler spec:

```bash
uv run python -m dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app review-outputs \
  --study-dir docs/studies/retron_hairpin_design \
  --materialized-root docs/studies/retron_hairpin_design/workbench/outputs/teto_pwm_trim_rescue_v1/materialized \
  --out-dir docs/studies/retron_hairpin_design/workbench/outputs/teto_pwm_trim_rescue_v1 \
  --format json
```

Expected review files:

- `reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.svg`
- `reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.png`
- `reviews/video/stills/01_t26-w00-19.png` through the ninth compact variant
  still
- `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4`
- `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.manifest.json`
- `reviews/handoff/teto_pwm_trim_rescue_v1.handoff.tsv`
- `reviews/handoff/teto-pwm-trim-rescue-v1.handoff.md`
- `reviews/review_manifest.json`

### Visual Contract

The PWM triptych is a 19 nt monotypic TetR elite view: every panel keeps a pale
full-site backdrop, retained payload positions are base-colored, trimmed-out
positions are light gray, and quiet retained-edge tick labels mark the active
span instead of generic coordinate ticks. It renders the plus-strand motif
occurrence at `[0,17)` and the minus-strand occurrence at `[2,19)` as separate
logo layers. Compact subtitles report only the nt count, retained span, and
rounded information content. The active triptych uses full 19 nt, mild 15 nt,
and stronger 13 nt payloads selected by the dual-site sliding-window IC rule.
The video consumes compact variant stills named like `t26-w02-17`, while
construct ids, MSD ids, source composition plots, folding status, and
reverse-complement evidence stay in the manifests.

### Implementation Boundary

Implementation ownership mirrors these artifact families:

- `review_outputs/contracts/`: deliverable-plan parsing and review manifest
  writing.
- `review_outputs/pwm/`: BaseRender-style PWM triptych rendering.
- `review_outputs/sequence/`: materialized sequence-index and evidence checks.
- `review_outputs/video/`: semantic stills and montage video.
- `review_outputs/handoff/`: sequence-handoff TSV and Markdown indexes.

The public facade remains `review_outputs/service.py`; the CLI calls that
facade instead of importing the individual renderer packages.

### Lifecycle

1. Persistent meaning lives in `../design_sets/`.
2. Executable sequence inputs live in `../../compiler/inputs/`.
3. Deliverable expectations live here.
4. Compact run evidence lives in `../provenance/`.
5. Bulky generated artifacts live in ignored `../outputs/` by default, or in an
   explicit transient output root when a caller needs isolation.
6. Reader SPOP evidence is added only after the experiment has run; it is not
   produced by this deliverable lane.

### Boundary

Do not put GenBank files, PNG/SVG review panels, videos, or compiled catalogs in
this directory. This lane holds durable contracts and reviewer-facing maps. The
actual files are emitted by the compiler/materializer or by a future
hypothesis-specific review renderer into explicit output roots.
