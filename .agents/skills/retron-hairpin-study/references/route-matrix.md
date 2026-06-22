# Route Matrix

Use this matrix to route Retron MSD compiler tasks without defaulting to
study-phase reporting.

| User question | Primary surface | Why |
| --- | --- | --- |
| "Here is an MSD ID." | `msd-design-references.md` | Complete labels should lint/compile directly. |
| "I have payload, cap, left base, and right base." | `msd-design-references.md`, then compiler CLI | User supplied reference parts; do not run solvers unless validation fails or metadata is missing. |
| "Here is a compiler spec or solved primitive rank." | `msd-design-references.md`, then `lint --spec` | Specs parse explicit parts and public primitive selectors without depending on manual label syntax. |
| "Here is a tetO trim spec." | `workbench/design_sets/`, then `lint --spec` | Payload trimming is source-backed design-set meaning; the compiler consumes literal sequences and typed metadata. |
| "Why are these variants in the experiment?" | `docs/studies/retron_hairpin_design/workbench/` | Persistent hypotheses, effect tags, and design-set membership live in the workbench, not generated compiler output. |
| "Where do the tetO PWM panel, sequence stills, video, and GenBank handoff outputs belong?" | `workbench/deliverables/`, then materialize and `review-outputs` | Deliverable expectations are persistent study contracts; bulky renders and exports remain generated output. |
| "Generate the tetO trim PWM triptych and sequence montage." | `review-outputs` against the materialized bundle | The review renderer consumes `sequence_index.tsv`; it does not scan generated directories ad hoc. |
| "I have left/right bases but no profile." | compiler parser | The profile is derived from bases and fails fast if `S0` is not `M`. |
| "I have a profile but not bases." | scar-nick route in `routes/README.md` / `routes/product/scar-nick-base-junction.md` | Base feasibility is a primitive search problem. |
| "I need a cap or shortening geometry." | released-product Snapback route in `routes/README.md` / `routes/product/released-product-snapback.md` | Cap/shortening is solved by Snapback, not the compiler. |
| "I need mismatch/boundary illustration." | YIU route in `routes/README.md` / `routes/quality/yiu-boundary-check.md` | YIU is contrast-only. |
| "I need sequence, SVG, PNG, or GenBank for selected parts." | compile reference, then materialize single-unit sequence bundle | The compiler emits one MSD unit per design and does not accept a repeat count. |
| "Open a transient Finder window with these outputs/deliverables." | materialize single-unit sequence bundle, then open the bundle root | Finder/output wording implies concrete artifacts; catalog JSONs alone are not the requested deliverable. |
| "Where is the Retron study status?" | `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json` | Use only for progress/history questions. |
| "What blocks the Retron study run?" | `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json` | Use only for blocker/readiness questions. |

## Compiler-First Boundary

- Start with input completeness, not study phase.
- If all required parts are present, validate and compile.
- Preserve exact user-specified labels. Checked-in cohorts are examples or named
  study fixtures, not replacements for live user input.
- If a part is missing, route to the smallest primitive solver.
- If sequence or visual artifacts are requested, materialize one MSD unit per
  design and keep complete-unit repeat expansion out of the compiler.
- If Finder/output/deliverable language is present, verify
  `manifest/indexes/sequence_index.tsv`, per-design GenBank files,
  `plots/secondary_structure.native.png`, `plots/composition_overview.svg`, and
  `plots/composition_overview.png` before reporting success.
- If a tetO trim review package is requested after materialization, run
  `review-outputs` into `workbench/outputs/teto_pwm_trim_rescue_v1/` and verify
  `reviews/review_manifest.json`, the logo-style PWM triptych, nine semantic
  still PNGs, the montage MP4, the montage manifest, the six-file
  `benchling_genbank/` import folder, and reverse-complement plus folding
  evidence.
- If primitive sources select multiple ranks, fail fast unless a future
  expansion contract is explicit; the preferred product surface is one selected
  cap rank plus one selected stem-base rank per design.
- If payload-trim metadata is present, keep it attached to literal payload
  sequences and reference rows; do not make the compiler infer payload sequence
  from PWM labels.
- Use `operations/runtime/command-groups/pipeline.yaml` only when a machine-readable command group is needed.
- Use `workbench/` when the question is persistent provenance rather than
  transient output generation.
- Use `workbench/deliverables/` when the question is the expected PWM,
  sequence-review, video, GenBank, review manifest, or outcome-overlay artifact
  map for a hypothesis.
- Do not reconstruct compiler behavior from generic Cruncher docs.
