# External Sources

Use the checked-in repo contracts below before running the materializer. Web
retrieval is not required for normal routing.

| Source | Owner | Checked | Purpose |
| --- | --- | --- | --- |
| `docs/studies/rt_lnrna_sponging_construct_triage/routes/reader-spop-condition-structure-matrix.md` | RT-lnRNA study | 2026-07-07 | First-hop route for this plot family. |
| `docs/studies/rt_lnrna_sponging_construct_triage/contexts/reader-spop-label-contract.md` | RT-lnRNA study | 2026-07-07 | Reader SPOP bridge semantics and positive-control policy. |
| `docs/studies/retron_hairpin_design/workbench/outputs/teto_pwm_trim_rescue_v1/reviews/review_manifest.json` | Retron hairpin study | 2026-07-07 | MSD structure source manifest. |
| `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/materialize.py` | RT-lnRNA study | 2026-07-07 | Materializer entrypoint. |
| `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/conditions.py` | RT-lnRNA study | 2026-07-07 | Condition labels, role constants, and heatmap column order. |
| `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/condition_matrix.py` | RT-lnRNA study | 2026-07-07 | Condition-long table builder. |
| `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/tables.py` | RT-lnRNA study | 2026-07-07 | Parquet schema and table writer. |
| `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/structure_manifest.py` | RT-lnRNA study | 2026-07-07 | Structure thumbnail manifest builder. |
| `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/render.py` | RT-lnRNA study | 2026-07-07 | Heatmap renderer. |

Freshness check: rerun the route command and tests when Reader experiments,
hairpin materialized outputs, or catalog metadata change.
