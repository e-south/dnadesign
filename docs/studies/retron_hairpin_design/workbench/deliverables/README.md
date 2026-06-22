## Retron Workbench Deliverables

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-22

This lane records hypothesis-specific review and handoff deliverables. It is
not a generated-output directory. Use it to answer what a study cohort should
emit, which source owns each artifact, and where a naive agent should look
before running or interpreting output generation.

### Records

- `teto_pwm_trim_rescue_v1.yaml`: deliverable plan for the nine-design
  bidirectional TetR PWM trim pilot. It links the design set, compiler spec,
  PWM provenance, materialized sequence bundle, PWM trim triptych, sequence
  montage video, review manifest, and GenBank handoff surfaces.

### tetO Trim Review Outputs

Open these first when reviewing the generated package:

1. `reviews/handoff/teto-pwm-trim-rescue-v1.handoff.md`
2. `reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.png`
3. `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4`
4. `reviews/review_manifest.json`

The handoff Markdown is the entry point. It maps compact variant ids such as
`r26-w02-17` to the generated GenBank, FASTA, and feature CSV files. Complete
metadata remains in `sequence_index.tsv` and `review_manifest.json`.

For Benchling import, use only:

```text
benchling_genbank/
```

That folder is intentionally flat: six reverse-complement GenBank files for
new trim variants `pES-retron-195` through `pES-retron-200`. The full parent
rows remain in the review bundle and are not copied into the import folder.
The assigned pES-retron ids, source precedent ids, and included trim rows are
declared in the deliverable plan, then consumed by the review renderer.

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
- `reviews/video/stills/01_pES-retron-26_tetO-w00-19.png`
- `reviews/video/stills/02_pES-retron-195_tetO-w02-17.png`
- `reviews/video/stills/03_pES-retron-196_tetO-w03-16.png`
- `reviews/video/stills/04_pES-retron-43_tetO-w00-19.png`
- `reviews/video/stills/05_pES-retron-197_tetO-w02-17.png`
- `reviews/video/stills/06_pES-retron-198_tetO-w03-16.png`
- `reviews/video/stills/07_pES-retron-180_tetO-w00-19.png`
- `reviews/video/stills/08_pES-retron-199_tetO-w02-17.png`
- `reviews/video/stills/09_pES-retron-200_tetO-w03-16.png`
- `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4`
- `reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.manifest.json`
- `reviews/handoff/teto_pwm_trim_rescue_v1.handoff.tsv`
- `reviews/handoff/teto-pwm-trim-rescue-v1.handoff.md`
- `reviews/review_manifest.json`

Expected Benchling import files, all reverse-complement GenBank records:

- `benchling_genbank/pES-retron-195-msd[TetR]-r26-w02-17.gb`
- `benchling_genbank/pES-retron-196-msd[TetR]-r26-w03-16.gb`
- `benchling_genbank/pES-retron-197-msd[TetR]-r43-w02-17.gb`
- `benchling_genbank/pES-retron-198-msd[TetR]-r43-w03-16.gb`
- `benchling_genbank/pES-retron-199-msd[TetR]-r180-w02-17.gb`
- `benchling_genbank/pES-retron-200-msd[TetR]-r180-w03-16.gb`

### Visual Contract

The PWM triptych is a 19 nt monotypic TetR elite view: every panel keeps a pale
full-site backdrop, retained payload positions are base-colored, trimmed-out
positions are light gray, and retained-edge cut lines mark the active span
instead of generic coordinate ticks. It renders the plus-strand motif
occurrence at `[0,17)` and the minus-strand occurrence at `[2,19)` as separate
logo layers. Compact subtitles report only the nt count, retained span, and
rounded information content. The active triptych uses full 19 nt, mild 15 nt,
and stronger 13 nt payloads selected by the dual-site sliding-window IC rule.
The PNG is rendered at no less than 3000 x 800 px; the SVG carries the same
metadata and remains the structured review source.
The video consumes review stills named with the canonical `pES-retron-*` id and
the retained tetO PWM window slug, while compact variant ids, source construct
ids, MSD ids, composition plots, folding status, and reverse-complement
evidence stay in the manifests.
The stills are 1920 x 1080 px images derived from materialized
`composition_overview.png` files. Their review titles use the
`pES-retron-XXX` names from `review_variant_ids` plus the retained tetO PWM
span, for example `tetO PWM [2,17)`. Subtitles report only scaffold and retained
payload length; control/target/candidate role metadata stays in the indexes.
The montage target is 1920 x 1080 px MP4.

### Implementation Boundary

Implementation ownership mirrors these artifact families:

- `review_outputs/contracts/`: deliverable-plan parsing and review manifest
  writing.
- `review_outputs/pwm/`: PWM triptych rendering through the public
  `dnadesign.baserender` API.
- `review_outputs/sequence/`: materialized sequence-index and evidence checks.
- `review_outputs/video/`: `pES-retron`-named stills and montage video.
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

Do not put GenBank files, PNG/SVG review panels, videos, or compiled catalogs
in this directory. This lane holds durable contracts and reviewer-facing maps.
Actual files are emitted by the compiler/materializer or review renderer into
explicit output roots.
