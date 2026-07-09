---
doc_id: study-retron-hairpin-design-workbench-deliverables
surface: study-workbench-deliverables
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-07-09
plane: handoff-plane
surface_role: deliverable-contracts
---

## Retron Workbench Deliverables

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-09

Records hypothesis-specific review and handoff deliverables. It is not a
generated-output directory. Use it to answer what a study cohort should emit,
which source owns each artifact, and where a naive agent should look before
running or interpreting output generation.

### Records

- `teto_retained_span_trim_tetr_pwm_elite_v1.yaml`: deliverable plan for the nine-design
  bidirectional TetR PWM trim pilot. It links the design set, compiler spec,
  PWM provenance, materialized sequence bundle, PWM trim triptych, sequence
  montage video, review manifest, and GenBank handoff surfaces. Its Benchling
  import records are MSD-only files named by `record_ids`, not whole-plasmid
  pES records.
- `teto_retained_span_trim_ecoli_working_v1.yaml`: deliverable plan for the six-design
  Eco1 tetO retained-span trim cohort. It assigns pES-retron-201 through pES-retron-206
  and keeps the same retained-span extents as the 195-200 TetR PWM pilot. Its
  PWM review panel also includes the untrimmed Eco1 tetO parent payload as a
  review-only baseline.

### tetO Trim Review Outputs

Open these first when reviewing the generated package:

1. `reviews/handoff/teto-retained-span-trim-tetr-pwm-elite-v1.handoff.md`
2. `reviews/pwm/teto_retained_span_trim_tetr_pwm_elite_v1.pwm_trim_triptych.png`
3. `reviews/video/teto_retained_span_trim_tetr_pwm_elite_v1.sequence_montage.mp4`
4. `reviews/review_manifest.json`

The handoff Markdown is the entry point. It maps compact variant ids such as
`r26-w02-17` to the generated GenBank, FASTA, and feature CSV files. Complete
metadata remains in `sequence_index.tsv` and `review_manifest.json`.

For Benchling import, use only:

```text
benchling_genbank/
```

That folder is intentionally flat: six reverse-complement GenBank files for
new trim variants. Filenames and LOCUS/ACCESSION values use `msd-retron-*`
record ids from the deliverable plan. Assigned `pES-retron-*` ids remain review
metadata. The full parent rows remain in the review bundle and are not copied
into the import folder.

### Generate

Preferred local output root:

```text
docs/studies/retron_hairpin_design/workbench/outputs/teto_retained_span_trim_tetr_pwm_elite_v1/
```

Recommended sequence-bundle root under that directory:

```text
materialized/
```

Generate the review package after materializing the nine-design compiler spec:

```bash
uv run python -m dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app review-outputs \
  --deliverable-plan docs/studies/retron_hairpin_design/workbench/deliverables/teto_retained_span_trim_tetr_pwm_elite_v1.yaml \
  --study-dir docs/studies/retron_hairpin_design \
  --materialized-root docs/studies/retron_hairpin_design/workbench/outputs/teto_retained_span_trim_tetr_pwm_elite_v1/materialized \
  --out-dir docs/studies/retron_hairpin_design/workbench/outputs/teto_retained_span_trim_tetr_pwm_elite_v1 \
  --format json
```

Expected review files:

- `reviews/pwm/teto_retained_span_trim_tetr_pwm_elite_v1.pwm_trim_triptych.svg`
- `reviews/pwm/teto_retained_span_trim_tetr_pwm_elite_v1.pwm_trim_triptych.png`
- `reviews/video/stills/01_pES-retron-26_tetO-w00-19.png`
- `reviews/video/stills/02_pES-retron-195_tetO-w02-17.png`
- `reviews/video/stills/03_pES-retron-196_tetO-w03-16.png`
- `reviews/video/stills/04_pES-retron-43_tetO-w00-19.png`
- `reviews/video/stills/05_pES-retron-197_tetO-w02-17.png`
- `reviews/video/stills/06_pES-retron-198_tetO-w03-16.png`
- `reviews/video/stills/07_pES-retron-180_tetO-w00-19.png`
- `reviews/video/stills/08_pES-retron-199_tetO-w02-17.png`
- `reviews/video/stills/09_pES-retron-200_tetO-w03-16.png`
- `reviews/video/teto_retained_span_trim_tetr_pwm_elite_v1.sequence_montage.mp4`
- `reviews/video/teto_retained_span_trim_tetr_pwm_elite_v1.sequence_montage.manifest.json`
- `reviews/handoff/teto_retained_span_trim_tetr_pwm_elite_v1.handoff.tsv`
- `reviews/handoff/teto-retained-span-trim-tetr-pwm-elite-v1.handoff.md`
- `reviews/review_manifest.json`

Expected Benchling import files, all reverse-complement GenBank records:

- `benchling_genbank/msd-retron-195.gb`
- `benchling_genbank/msd-retron-196.gb`
- `benchling_genbank/msd-retron-197.gb`
- `benchling_genbank/msd-retron-198.gb`
- `benchling_genbank/msd-retron-199.gb`
- `benchling_genbank/msd-retron-200.gb`

### Eco1 tetO Retained-Span Trim Outputs

Open the Eco1 retained-span deliverable plan first:

```text
workbench/deliverables/teto_retained_span_trim_ecoli_working_v1.yaml
```

That plan assigns `pES-retron-201` through `pES-retron-206` for review
metadata and declares separate `record_ids` for the MSD-only GenBank records.
It keeps the retained spans at `[2,17)` and `[3,16)`. This cohort changes the
payload family, not the trim extent. The PWM triptych has three panels: full
Eco1 tetO parent payload, 15 nt trim, and 13 nt trim. The full panel is a review
baseline and is not copied into the Benchling import folder. The
`benchling_genbank_import.descriptions` mapping is the source for the concise
GenBank definition text and Benchling index prose.

Generate the review package after materializing
`../../compiler/inputs/teto_retained_span_trim_ecoli_working_v1.spec.yaml`:

```bash
uv run python -m dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app review-outputs \
  --deliverable-plan docs/studies/retron_hairpin_design/workbench/deliverables/teto_retained_span_trim_ecoli_working_v1.yaml \
  --materialized-root docs/studies/retron_hairpin_design/workbench/outputs/teto_retained_span_trim_ecoli_working_v1/materialized \
  --out-dir docs/studies/retron_hairpin_design/workbench/outputs/teto_retained_span_trim_ecoli_working_v1 \
  --format json
```

Expected Benchling import files, all reverse-complement GenBank records:

- `benchling_genbank/msd-retron-201.gb`
- `benchling_genbank/msd-retron-202.gb`
- `benchling_genbank/msd-retron-205.gb`
- `benchling_genbank/msd-retron-206.gb`
- `benchling_genbank/msd-retron-203.gb`
- `benchling_genbank/msd-retron-204.gb`

Expected PWM review files:

- `reviews/pwm/teto_retained_span_trim_ecoli_working_v1.pwm_trim_triptych.svg`
- `reviews/pwm/teto_retained_span_trim_ecoli_working_v1.pwm_trim_triptych.png`

### Visual Contract

The PWM triptych renders one 19 nt parent payload coordinate system per
deliverable. Every panel keeps a pale full-site backdrop, base-colors retained
payload positions, dims trimmed-out positions, and uses retained-edge cut lines
instead of generic coordinate ticks. The TetR PWM elite cohort renders plus and minus
motif occurrences at `[0,17)` and `[2,19)`. The Eco1 tetO cohort renders both motif
orientations at `[1,18)`. Compact subtitles report only nt count, retained
span, and rounded information content. The active panels are full 19 nt, mild
15 nt, and stronger 13 nt payload views. A full panel can be review-only when
the deliverable assigns only trimmed constructs. The SVG records
`data-requires-materialized-sequence` for each panel.
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
- `review_outputs/contracts/record_ids.py`: MSD-only Benchling record-id
  validation.
- `review_outputs/handoff/`: sequence-handoff TSV/Markdown indexes and
  Benchling GenBank import writing.
- `review_outputs/handoff/genbank_features.py`: reverse-complement GenBank
  feature-direction normalization.

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
in this directory. Durable contracts and reviewer-facing maps belong here.
Actual files are emitted by the compiler/materializer or review renderer into
explicit output roots.
