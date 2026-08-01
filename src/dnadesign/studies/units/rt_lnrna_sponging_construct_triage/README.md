# RT-lnRNA Sponging Construct Triage

Study-owned helpers for the checked-in `docs/studies/rt_lnrna_sponging_construct_triage`
contract record.

This package owns validation around study fixtures, source-authority records,
the study-level Construct materialization proof for the two checked-in control
candidates, the exact Reader-evidence-to-compositional-subject binding, and the
six-view representation handoff contract.

Materialize an evidence-binding artifact from one verified Reader experiment
with:

```bash
uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize \
  --reader-root ../reader \
  --experiment-route-registry ../.agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json \
  --experiment-id 20260727_retron_Eco1_26_D01_D02_P01_P03_DP01_DP03_benchmark \
  --output /tmp/retron-reader-evidence-bindings.json
```

The command first requires the experiment to be an exact selected member of
the bridge-owned `rt_competence_subject_binding` route and requires that route
to pass its live Reader readiness check. Only then does it resolve the exact
`experiments/YYYY/<experiment-id>` path, verify the Reader catalog-v4 and
schema-v6 dataframe contract plus exact revision, revision digest, and content
digest. This separately routed validation experiment must expose its
source-declared biological replicate identity
(`replicate_kind=biological`,
`replicate_identity_field=biological_replicate_id`). The binding emits identity
and provenance fields without measurements or scientific interpretations.

It does not run Evo2, LatentDNA, or OPAL workloads, and it does not fabricate
feature sidecars.

The study-owned `reporter_response/metastudy` package separately materializes
descriptive profiles from its eight admitted kinetic records and bindings. Its exact
condition ontology maps Reader treatment labels to baseline, positive-control,
and dose roles. Candidate selection remains 500-uM-only; explicitly present
5-uM and 50-uM conditions appear only in endpoint and centered-window
sensitivity profiles. Exact ten-minute cadence, aligned channels, three
within-acquisition observations per stratum, finite values, and quality provenance
are fail-closed requirements. Experiment, plate, sheet, well, and position are
acquisition provenance, not replicate identities. A declared replicate field
is the only source of biological-replicate identity; absent that field, the
identity remains unknown. Observations are median-reduced within an acquisition
before cross-acquisition selection and leave-one-acquisition-out evaluation.
