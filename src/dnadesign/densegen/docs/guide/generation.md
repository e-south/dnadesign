# Generation and constraints

DenseGen generates a **plan** of sequences using dense-arrays. Each plan item is a constraint bucket.

## Plan definition

- Each plan item has a `name` and **either** `quota` or `fraction`.
- Mixing quotas and fractions across items is not allowed.

```yaml
plan:
  - name: sigma70
    quota: 200
    required_regulators: ["LexA", "CpxR"]
    min_required_regulators: 2
    min_count_by_regulator:
      LexA: 2
    fixed_elements:
      promoter_constraints:
        - upstream: "TTGACA"
          downstream: "TATAAT"
          spacer_length: [16, 18]
```

## Promoter constraints

Use `fixed_elements.promoter_constraints` to enforce fixed motifs and spacing. Motifs must be A/C/G/T only.

Fields:
- `upstream`, `downstream` (motif strings)
- `spacer_length: [min, max]`
- `upstream_pos`, `downstream_pos` (optional ranges)

## Side biases (positional preferences)

Use `fixed_elements.side_biases` to bias motifs to the left or right of the sequence.

```yaml
fixed_elements:
  side_biases:
    left: ["GAAATAACATAATTGA", "TTATATTTTACCCATTT"]
    right: ["CATAAGAAAAA", "CATTCATTTG"]
```

Rules:
- Motifs must be A/C/G/T only.
- Motifs **must exist in the sampled library** (DenseGen fails fast if missing).
- A motif cannot be in both `left` and `right`.

## Solver strategy

DenseGen exposes dense-arrays solution modes via `solver.strategy`:

- `iterate` — yield solutions in descending score.
- `diverse` — yield solutions with diversity-biased ordering.
- `optimal` — only the best solution per library.
- `approximate` — heuristic solution per library (no solver options; backend optional).
- `strands` — `single | double` (default: `double`).

```yaml
solver:
  backend: CBC
  strategy: diverse
  options: ["Threads=8", "TimeLimit=10"]
  strands: double
```

## Sampling controls

`generation.sampling` controls how binding-site libraries are built (pool strategy, coverage, uniqueness, caps, and relaxation). DenseGen records sampling policy and outcomes in metadata.

Key fields:
- `pool_strategy`: `full | subsample | iterative_subsample`
- `library_size` (used for subsample strategies)
- `cover_all_regulators`, `unique_binding_sites`, `max_sites_per_regulator`
- `iterative_max_libraries`, `iterative_min_new_solutions`

Notes:
- `pool_strategy: full` and `subsample` use a single library (no resampling).
- Use `iterative_subsample` when you want multiple libraries or adaptive retries.
- `unique_binding_sites` enforces uniqueness at the regulator+sequence pair level.

## Regulator constraints

DenseGen supports three regulator constraint modes per plan item:

- `required_regulators`: enforce at least one site per listed regulator.
- `min_required_regulators`: enforce at least K distinct regulators (k‑of‑n).
- `min_count_by_regulator`: enforce per‑regulator minimum counts.

Solver strategies (`iterate|diverse|optimal`) enforce these constraints at the solver level;
`approximate` runs are validated in the pipeline to keep behavior consistent.

Notes:
- `min_count_by_regulator` takes precedence over the global `runtime.min_count_per_tf`
  for listed regulators (DenseGen uses the maximum of the two).
