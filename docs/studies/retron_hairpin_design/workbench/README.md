---
doc_id: study-retron-hairpin-design-workbench
surface: study-workbench
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-07-09
plane: intent-plane
surface_role: study-workbench-router
---

## Retron Experimental Workbench

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-09

This workbench is the persistent record for Retron experimental intent. It sits
between the one-hop study router and transient compiler outputs.

Use it for:

- durable design-set membership;
- hypothesis, effect, and comparator tags;
- compiler invocation provenance;
- materialization provenance when GenBank, plot, or Reader-facing bundles are
  emitted.
- local review-output roots for PWM panels, sequence montage videos, and review
  manifests under ignored `outputs/`.

Do not use it for:

- generated compiler bundles as durable records;
- Cruncher, Construct, Folding, or BaseRender internals;
- generic tool runbooks.

### Layout

```text
workbench/
  ontology/
    directions.yaml
    feature_roles.yaml
    payload_binding_sites.yaml
  design_sets/
    scar_nick_profile_panel_v1.yaml
    teto_retained_span_trim_tetr_pwm_elite_v1.yaml
    teto_retained_span_trim_ecoli_working_v1.yaml
  deliverables/
    teto_retained_span_trim_tetr_pwm_elite_v1.yaml
    teto_retained_span_trim_ecoli_working_v1.yaml
  provenance/
    compiler_runs/
      2026-05-18-msd-177-194.compile.yaml
    materializations/
      2026-05-18-msd-177-194.single-unit.yaml
  outputs/        # generated and gitignored
```

### Route By Question

| Need | Open |
| --- | --- |
| Direction, payload, and feature-role vocabulary | [ontology/](ontology/README.md) |
| Authoritative cohort membership | [design_sets/](design_sets/README.md) |
| Hypothesis-specific review and handoff deliverables | [deliverables/](deliverables/README.md) |
| Compiler and materialization run records | [provenance/](provenance/README.md) |
| Local generated review bundles | `outputs/` |

### Boundary

The workbench owns study meaning: why these variants belong together, which
experimental directions they test, and which review or handoff deliverables are
expected. The study compiler owns validation and transient
catalog/materialization output. Sibling tools own their primitive or
artifact-service behavior.

`../compiler/inputs/msd_design_hit_labels.txt` remains a convenience compiler input.
The design set under `design_sets/` is the authoritative cohort record.
Generated review bundles may live under ignored `outputs/` for local inspection,
but provenance in `provenance/materializations/` and contracts in
`deliverables/` are the durable records.
