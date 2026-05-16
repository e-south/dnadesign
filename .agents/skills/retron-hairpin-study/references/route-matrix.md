# Route Matrix

Use this matrix to route Retron MSD compiler tasks without defaulting to
study-phase reporting.

| User question | Primary surface | Why |
| --- | --- | --- |
| "Here is an MSD ID." | `msd-design-references.md` | Complete labels should lint/compile directly. |
| "I have payload, cap, left base, and right base." | `msd-design-references.md`, then compiler CLI | User supplied reference parts; do not run solvers unless validation fails or metadata is missing. |
| "Here is a compiler spec or solved primitive rank." | `msd-design-references.md`, then `lint --spec` | Specs parse explicit parts and public primitive selectors without depending on manual label syntax. |
| "I have left/right bases but no profile." | compiler parser | The profile is derived from bases and fails fast if `S0` is not `M`. |
| "I have a profile but not bases." | scar-nick route in `routes.md` | Base feasibility is a primitive search problem. |
| "I need a cap or shortening geometry." | released-product Snapback route in `routes.md` | Cap/shortening is solved by Snapback, not the compiler. |
| "I need mismatch/boundary illustration." | YIU route in `routes.md` | YIU is contrast-only. |
| "I need sequence, SVG, PNG, or GenBank for selected parts." | compile reference, then materialize single-unit sequence bundle | The compiler emits one MSD unit per design and does not accept a repeat count. |
| "Open a transient Finder window with these outputs/deliverables." | materialize single-unit sequence bundle, then open the bundle root | Finder/output wording implies concrete artifacts; catalog JSONs alone are not the requested deliverable. |
| "Where is the old study status?" | `cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json` | Use only for progress/history questions. |
| "What blocks a legacy study run?" | `cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json` | Use only for blocker/readiness questions. |

## Compiler-First Boundary

- Start with input completeness, not study phase.
- If all required parts are present, validate and compile.
- If a part is missing, route to the smallest primitive solver.
- If sequence or visual artifacts are requested, materialize one MSD unit per
  design and keep complete-unit repeat expansion out of the compiler.
- If Finder/output/deliverable language is present, verify
  `manifest/indexes/sequence_index.tsv`, per-design GenBank files,
  `plots/secondary_structure.native.png`, `plots/composition_overview.svg`, and
  `plots/composition_overview.png` before reporting success.
- If primitive sources select multiple ranks, fail fast unless a future
  expansion contract is explicit; the preferred product surface is one selected
  cap rank plus one selected stem-base rank per design.
- Use `pipeline.yaml` only when a machine-readable command group is needed.
- Do not reconstruct compiler behavior from generic Cruncher docs.
