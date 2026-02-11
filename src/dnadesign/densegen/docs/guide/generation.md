## Generation (Stage-B + constraints)

This guide explains how plan config drives runtime behavior once Stage-A pools exist.

Use [../reference/config.md](../reference/config.md) for exact field names.

### Mental model

DenseGen generation is a sequence of subprocesses:

1. **Stage-A** creates input pools (candidate TFBS per input).
2. **Stage-B** samples a plan-scoped library from those pools.
3. **Solver** places sampled sites into a sequence under plan constraints.
4. **Quota loop** repeats until each plan reaches `quota` or runtime limits are hit.

The generation plan is the contract for steps 2-4.

### What the plan controls

- which inputs feed each plan (`sampling.include_inputs`)
- how many accepted arrays each plan must produce (`quota`)
- what regulators must appear (`regulator_constraints`)
- fixed sequence structure (`fixed_elements`, for example promoters)

### Minimal plan example (with intent)

```yaml
plan:
  # Generate 200 accepted arrays for this plan.
  - name: sigma70
    quota: 200
    sampling:
      # Build this plan's Stage-B library from these Stage-A inputs.
      include_inputs: [lexA_pwm, cpxR_pwm, background]
    regulator_constraints:
      groups:
        # Require both regulators in each accepted solution.
        - name: core
          members: ["LexA", "CpxR"]
          min_required: 2
      min_count_by_regulator:
        # Also require at least one placement per regulator.
        LexA: 1
        CpxR: 1
```

### Key rules

- `generation.sequence_length` must fit fixed elements plus sampled sites.
- If sequence length is too short, infeasibility is recorded in diagnostics.
- Regulator names in constraints must match Stage-A `tf` labels exactly.
- `include_inputs` should align with the biological intent of each plan; over-broad input sets reduce controllability.

### Fixed elements

Common fixed constraints:

- promoter constraints (fixed motif + spacer layout)
- side biases (prefer left or right side for selected motifs)

```yaml
fixed_elements:
  promoter_constraints:
    - upstream: "TTGACA"
      downstream: "TATAAT"
      spacer_length: [15, 19]
  side_biases:
    # Motifs listed here are placement preferences, not hard guarantees.
    left: ["GAAATAACATAATTGA"]
    right: ["CATAAGAAAAA"]
```

Motifs must be A/C/G/T. Invalid motifs and impossible fixed layouts fail fast.

### Solver strategy

DenseGen exposes dense-arrays solving strategies through `solver.strategy`:

- `iterate`
- `diverse`
- `optimal`
- `approximate`

```yaml
solver:
  backend: CBC
  strategy: diverse
  time_limit_seconds: 10
```

If backend availability does not match config, `dense validate-config --probe-solver` fails fast.

### Stage-level debugging checklist

Use this mapping when a plan underperforms:

1. **Stage-A issue** (pool quality/coverage):
   - inspect `outputs/pools/*` and Stage-A summary plots
2. **Stage-B issue** (library composition):
   - inspect `outputs/libraries/*` and `dense inspect run --library`
3. **Solver/runtime issue** (rejects, duplicates, failures):
   - inspect `outputs/tables/attempts.parquet` and run-health plots
4. **Output sink issue** (Parquet/USR mismatch or event flow):
   - inspect `outputs/meta/run_manifest.json` plus sink-specific artifacts

---

@e-south
