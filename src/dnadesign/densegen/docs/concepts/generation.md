## Generation model

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This concept page explains how DenseGen turns Stage-A pools into accepted sequences under plan constraints. Read it when you need to reason about quotas, fixed elements, and solver feasibility before editing generation config. For exact field definitions, use the **[config reference](../reference/config.md)**.

### What plans control
This section describes the generation-level contract each plan contributes to runtime behavior.

- `sampling.include_inputs` selects which Stage-A pools feed the plan library.
- `sequences` sets the per-plan milestone target for accepted sequences.
- `regulator_constraints` sets minimum regulator presence rules.
- `fixed_elements` sets hard sequence geometry such as promoter elements.

### Minimal plan shape
This section shows the smallest useful plan pattern and labels the intent of each key.

```yaml
plan:
  - name: sigma70_demo
    sequences: 24
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

- `promoter_constraints` and `fixed_element_matrix` are hard geometry constraints.
- `side_biases` are placement preferences and may not be satisfiable in all solutions.
- Motifs must be valid DNA alphabet (`A`, `C`, `G`, `T`) and infeasible geometry fails fast.

### Fixed-element matrix expansion model
This section explains the deterministic config-time expansion model behind `fixed_element_matrix`.

#### Intent
`fixed_element_matrix` lets one logical plan compile into many explicit concrete plans at config load time. The goal is compact configs with explicit, reproducible runtime behavior.

#### Intended use cases
- Combinatorial core/promoter panel studies without hand-writing many plans.
- Stress-condition sweeps where one core element is fixed while a curated variant set is scanned.
- Mixed workflows where matrix-expanded plans and plain non-expanded plans coexist in the same `generation.plan[]`.

#### Core functionality
- Defined at `generation.plan[].fixed_elements.fixed_element_matrix`.
- Pulls upstream/downstream variants from named motif sets.
- Supports pairing modes:
  - `zip`: pair by shared variant IDs.
  - `cross_product`: Cartesian product across selected upstream and downstream IDs.
  - `explicit_pairs`: use only listed `(up, down)` pairs.
- Optional selectors `upstream_variant_ids` and `downstream_variant_ids` constrain the expansion domain.
- Optional `expanded_name_template` controls expanded plan names with placeholders like `{base}`, `{up}`, `{down}`, `{up_seq}`, `{down_seq}`.

#### Math and quota operations
- Expansion cardinality per plan:
  - `zip`: number of matching IDs.
  - `cross_product`: `|U| * |D|`.
  - `explicit_pairs`: number of configured pairs.
- Quota contract:
  - Every plan must define `sequences > 0`.
  - For matrix plans, `sequences` is distributed across expanded variants as evenly as possible.
  - Distribution uses expansion order: each variant receives `floor(sequences / variants)` and the first `sequences % variants` variants receive one extra.
  - If `sequences < variants`, only the first `sequences` variants are active (quota `1` each).
- Global guardrails are enforced after expansion:
  - `generation.expansion.max_plans`

#### Lifecycle
1. Parse YAML and validate schema.
2. Normalize motif-set structures.
3. Expand `generation.plan[]` deterministically into concrete plans.
4. Enforce uniqueness, geometry feasibility, quotas, and global expansion caps.
5. Validate motif-set references used by sequence-constraint and background forbid rules.
6. Run Stage-B using resolved concrete plans only.

#### Policy and behavior guarantees
- Fail-fast only: unknown motif sets, invalid variant IDs, pairing mismatches, quota math errors, duplicate expanded names, and cap overflow all hard-fail validation.
- No adaptive runtime expansion: the expansion set is fixed at config load.
- Reproducible compilation: the same config yields the same expanded plans and quotas.
- Motif exclusion can use motif-set-derived rules in both background filters and global sequence constraints to reduce drift.

#### Current packaged workspace behavior
- `demo_tfbs_baseline` and `demo_sampling_baseline` do not use matrix expansion.
- `study_constitutive_sigma_panel` uses matrix expansion for a full panel:
  - `sigma70_panel`: `6 x 8 = 48` variants, `sequences: 100`.
  - Total: `48` concrete plans, aggregate target `100` (`4` variants at quota `3`, `44` variants at quota `2`).
- `study_stress_ethanol_cipro` uses curated upstream variants with fixed downstream consensus:
  - Three base plans (`ethanol`, `ciprofloxacin`, `ethanol_ciprofloxacin`) each expand to `5` variants.
  - Total: `15` concrete plans.
  - Uniform per-variant quotas: `60/5=12`, `60/5=12`, `80/5=16`.

### How sequence constraints work
This section explains how global sequence constraints apply after sequence assembly.

- `sequence_constraints.forbid_kmers` can enforce strand-aware motif exclusion.
- `sequence_constraints.allowlist` defines explicit fixed-element exceptions.
- Final validation occurs on assembled sequences, so join-spanning and pad-created motifs are checked.

### Solver settings and limits
This section describes how solver settings bound runtime behavior.

- `solver.backend` chooses the installed backend (`CBC` or `GUROBI`).
- `solver.strategy` selects dense-arrays solve strategy.
- `solver.solver_attempt_timeout_seconds` caps per-attempt time.
- Runtime guardrails in `densegen.runtime` still apply even with permissive solver settings.

### Debugging generation failures
This section gives the shortest route to diagnose infeasible plans.

1. Validate config expansion and inspect resolved plans.
2. Inspect Stage-B library summaries for input starvation.
3. Inspect `outputs/meta/events.jsonl` for explicit rejection reasons.
4. Reduce plan complexity before increasing solver limits.

For command-level troubleshooting flow, use **[DenseGen CLI reference](../reference/cli.md)** and **[sampling model](sampling.md)**.
