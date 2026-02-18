## Generation model

This concept page explains how DenseGen turns Stage-A pools into accepted sequences under plan constraints. Read it when you need to reason about quotas, fixed elements, and solver feasibility before editing generation config. For exact field definitions, use the **[config reference](../reference/config.md)**.

### What plans control
This section describes the generation-level contract each plan contributes to runtime behavior.

- `sampling.include_inputs` selects which Stage-A pools feed the plan library.
- `quota` sets how many accepted sequences the plan must produce.
- `regulator_constraints` sets minimum regulator presence rules.
- `fixed_elements` sets hard sequence geometry such as promoter elements.

### Minimal plan shape
This section shows the smallest useful plan pattern and labels the intent of each key.

```yaml
plan:
  - name: sigma70_demo
    quota: 24
    sampling:
      include_inputs: [lexA_pwm, cpxR_pwm, background]
    regulator_constraints:
      groups:
        - name: core_response
          members: [lexA_CTGTATAWAWWHACA, cpxR_MANWWHTTTAM]
          min_required: 1
```

### How fixed elements work
This section clarifies what fixed-element keys enforce as hard constraints versus preferences.

- `promoter_constraints` and `promoter_matrix` are hard geometry constraints.
- `side_biases` are placement preferences and may not be satisfiable in all solutions.
- Motifs must be valid DNA alphabet (`A`, `C`, `G`, `T`) and infeasible geometry fails fast.

### How sequence constraints work
This section explains how global sequence constraints apply after sequence assembly.

- `sequence_constraints.forbid_kmers` can enforce strand-aware motif exclusion.
- `sequence_constraints.allowlist` defines explicit fixed-element exceptions.
- Final validation occurs on assembled sequences, so join-spanning and pad-created motifs are checked.

### Solver settings and limits
This section describes how solver settings bound runtime behavior.

- `solver.backend` chooses the installed backend (`CBC` or `GUROBI`).
- `solver.strategy` selects dense-arrays solve strategy.
- `solver.time_limit_seconds` caps per-attempt time.
- Runtime guardrails in `densegen.runtime` still apply even with permissive solver settings.

### Debugging generation failures
This section gives the shortest route to diagnose infeasible plans.

1. Validate config expansion and inspect resolved plans.
2. Inspect Stage-B library summaries for input starvation.
3. Inspect `outputs/meta/events.jsonl` for explicit rejection reasons.
4. Reduce plan complexity before increasing solver limits.

For command-level troubleshooting flow, use **[DenseGen CLI reference](../reference/cli.md)** and **[sampling model](sampling.md)**.
