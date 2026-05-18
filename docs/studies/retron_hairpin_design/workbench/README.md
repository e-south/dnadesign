## Retron Experimental Workbench

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This workbench is the persistent record for Retron experimental intent. It sits
between the one-hop study router and transient compiler outputs.

Use it for:

- durable design-set membership;
- hypothesis, effect, and comparator tags;
- compiler invocation provenance;
- materialization provenance when GenBank, plot, or Reader-facing bundles are
  emitted.

Do not use it for:

- generated compiler bundles;
- Cruncher, Construct, Folding, or BaseRender internals;
- generic tool runbooks.

### Layout

```text
workbench/
  ontology/
    directions.yaml
  design_sets/
    scar_nick_profile_panel_v1.yaml
  provenance/
    compiler_runs/
      2026-05-18-msd-177-194.compile.yaml
    materializations/
      2026-05-18-msd-177-194.single-unit.yaml
```

### Route By Question

| Need | Open |
| --- | --- |
| Direction and effect vocabulary | [ontology/](ontology/README.md) |
| Authoritative cohort membership | [design_sets/](design_sets/README.md) |
| Compiler and materialization run records | [provenance/](provenance/README.md) |

### Boundary

The workbench owns study meaning: why these variants belong together and which
experimental directions they test. The study compiler owns validation and
transient catalog/materialization output. Sibling tools own their primitive or
artifact-service behavior.

`../compiler/msd_design_hit_labels.txt` remains a convenience compiler input.
The design set under `design_sets/` is the authoritative cohort record.
