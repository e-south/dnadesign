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
  review manifest, and GenBank cloning handoff.

### tetO Trim Review Outputs

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
- `reviews/video/stills/01_control_retron26_TetR_full.png` through the ninth
  semantically named still
- `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4`
- `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.manifest.json`
- `reviews/review_manifest.json`

The PWM triptych is a 19 nt monotypic TetR elite view: every panel keeps a pale
full-site backdrop, retained payload positions are base-colored, trimmed-out
positions are light gray, and quiet retained-edge tick labels mark the active
span instead of generic coordinate ticks. It renders the plus-strand motif
occurrence at `[0,17)` and the minus-strand occurrence at `[2,19)` as separate
logo layers. Compact subtitles report only the nt count, retained span, and
rounded information content. The active triptych uses full 19 nt, mild 15 nt,
and stronger 12 nt payloads selected by the dual-site sliding-window IC rule.
The video consumes the semantically named review stills, while construct ids,
MSD ids, source composition plots, folding status, and reverse-complement
evidence stay in the manifests.

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
