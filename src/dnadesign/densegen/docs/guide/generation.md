## Generation (Stage-B + constraints)

Generation is driven by a **plan**. Each plan item defines how many sequences to build and which
regulators must appear. Stage-B sampling builds solver libraries from Stage-A pools, and the
dense-arrays solver assembles sequences subject to constraints.

Use `reference/config.md` for exact fields.

---

### Plan definition (minimal)

Each plan item has a `name` and either `quota` or `fraction`.

```yaml
plan:
  - name: sigma70
    quota: 200
    regulator_constraints:
      groups:
        - name: core
          members: ["LexA", "CpxR"]
          min_required: 2
      min_count_by_regulator:
        LexA: 1
        CpxR: 1
```

Notes:
- `generation.sequence_length` must fit all fixed elements and TFBS placements.
- Group members must match the `tf` labels in the Stage-A pools.

---

### Fixed elements

Two common constraint types:

- **Promoter constraints**: fixed motifs + spacing.
- **Side biases**: bias specific motifs toward left/right.

```yaml
fixed_elements:
  promoter_constraints:
    - upstream: "TTGACA"
      downstream: "TATAAT"
      spacer_length: [15, 19]
  side_biases:
    left: ["GAAATAACATAATTGA"]
    right: ["CATAAGAAAAA"]
```

Motifs must be A/C/G/T only. DenseGen fails fast if fixed motifs cannot be placed.

---

### Stage-B sampling (summary)

Stage-B sampling lives under `densegen.generation.sampling`.
It builds solver libraries from Stage-A pools and is the only stage that resamples during a run.

See `guide/sampling.md` for the Stage-B overview.

---

### Solver strategy (summary)

DenseGen exposes dense-arrays strategies via `solver.strategy`:
`iterate`, `diverse`, `optimal`, or `approximate`.

DenseGen fails fast if the chosen solver backend is unavailable.

```yaml
solver:
  backend: CBC
  strategy: diverse
  time_limit_seconds: 10
```

---

@e-south
