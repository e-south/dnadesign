![cluster banner](assets/cluster-banner.svg)

`cluster` is the exploratory downstream surface for unsupervised clustering, UMAP visualization, and related summaries over one chosen feature matrix.
It records reusable outputs in a run store under `results/` and stays decoupled from upstream feature generation.

See the [repository docs index](../../../docs/README.md) for cross-tool workflow routes and runbooks. For the authoritative cross-tool source-of-truth path that builds an infer-annotated promoter feature matrix before clustering, use [promoter characterization feature matrix](../usr/docs/operations/promoter-characterization-feature-matrix.md).

---

## Ownership boundary

- `cluster` is a downstream exploratory consumer. It does not generate `infer__...` columns or own the upstream USR handoff.
- Precondition: one explicit `infer__...` column is already present and chosen as `X` in the input dataset or file before you run `cluster fit`, `cluster umap`, or `cluster analyze`.
- For feature generation and durable dataset assembly, return to [promoter characterization feature matrix](../usr/docs/operations/promoter-characterization-feature-matrix.md), [infer docs](../infer/docs/README.md), or the [repository docs index](../../../docs/README.md).
- If you need supervised label/train/select rather than exploratory structure, switch to [USR dataset with infer-derived X -> OPAL active learning](../opal/docs/workflows/usr-infer-x-active-learning.md).

---

## Start here

- If you do not yet have one explicit `infer__...` column chosen as `X`, start with [promoter characterization feature matrix](../usr/docs/operations/promoter-characterization-feature-matrix.md) and return here after infer write-back is complete.
- If you already have that chosen `X` column and want the first runnable path, open [cluster docs by workflow](docs/README.md) and start with [exploratory clustering workflow](docs/workflows/exploratory-clustering.md).
- If you need command, preset, results, or OPAL-join semantics before running, use [cluster CLI contracts](docs/reference/cli-contracts.md).
- If you need the package-level ownership split versus USR and OPAL, use [cluster ownership boundary](docs/concepts/ownership-boundary.md).

## Task routes

- Run the first `fit -> umap -> analyze` pass for a chosen `X` column: [exploratory clustering workflow](docs/workflows/exploratory-clustering.md).
- Reuse OPAL outputs for exploratory plots or summaries: [cluster CLI contracts](docs/reference/cli-contracts.md#opal-join-contract).
- Inspect presets, jobs, and results layout before editing configs: [cluster CLI contracts](docs/reference/cli-contracts.md#jobs-presets-and-results-layout).
- Decide whether you should stay in `cluster` or switch to OPAL: [cluster ownership boundary](docs/concepts/ownership-boundary.md).

## Documentation map

- [cluster docs by workflow](docs/README.md): task-first router for clustering work once `X` already exists.
- [cluster docs by type](docs/index.md): workflow, concept, and reference split.
- [exploratory clustering workflow](docs/workflows/exploratory-clustering.md): first runnable `fit`, `umap`, `intra-sim`, and `analyze` path.
- [cluster CLI contracts](docs/reference/cli-contracts.md): command surface, jobs/presets layout, OPAL joins, env vars, and troubleshooting.
- [cluster ownership boundary](docs/concepts/ownership-boundary.md): downstream role relative to USR, infer, and OPAL.
- [repository docs index](../../../docs/README.md): cross-tool workflow routes.

---

## CLI surface

```bash
uv run cluster --help
uv run cluster fit --help
uv run cluster umap --help
uv run cluster analyze --help
uv run cluster intra-sim --help
uv run cluster sweep --help
uv run cluster delete-columns --help
```

## Results and artifacts

- Checked-in inputs live under `jobs/` and `presets/`.
- Generated outputs live under `results/`.
- For exact results, reuse, and cleanup semantics, see [cluster CLI contracts](docs/reference/cli-contracts.md#results-and-artifacts).
