![cluster banner](assets/cluster-banner.svg)

`cluster` is the exploratory downstream surface for unsupervised clustering, UMAP visualization, and related summaries over one chosen feature matrix from a USR dataset or a CSV/Parquet file.
It records reusable outputs in a workspace-scoped artifact root and stays decoupled from upstream feature generation.

See the [repository docs index](../../../docs/README.md) for cross-tool workflow routes and runbooks. For the authoritative cross-tool source-of-truth path that builds an infer-annotated promoter feature matrix before clustering, use [promoter characterization feature matrix](../usr/docs/operations/promoter-characterization-feature-matrix.md).

---

## Ownership boundary

- `cluster` is a downstream exploratory consumer. It does not generate upstream feature columns or own the upstream USR handoff.
- Precondition: one explicit chosen feature definition is already present before you run `cluster fit`, `cluster umap`, or `cluster analyze`.
- For feature generation and durable dataset assembly, return to [promoter characterization feature matrix](../usr/docs/operations/promoter-characterization-feature-matrix.md), [infer docs](../infer/docs/README.md), or the [repository docs index](../../../docs/README.md).
- If you need supervised label/train/select rather than exploratory structure, switch to [USR dataset with infer-derived X -> OPAL active learning](../opal/docs/workflows/usr-infer-x-active-learning.md).

---

## Start here

- If you do not yet have one explicit chosen feature definition, start with [promoter characterization feature matrix](../usr/docs/operations/promoter-characterization-feature-matrix.md) for the repository's canonical infer-backed route, then return here after feature write-back is complete.
- If you already have that chosen feature definition and want the first runnable path, open [cluster docs by workflow](docs/README.md) and start with [exploratory clustering workflow](docs/workflows/exploratory-clustering.md).
- If you need workspace, preset, results, or OPAL-join semantics before running, use [cluster CLI contracts](docs/reference/cli-contracts.md).
- If you need an in-process automation boundary instead of CLI shelling, use the public [`dnadesign.cluster.api`](api.py) ad hoc and workspace helpers. They execute the same shared runtime as the CLI and do not bounce through Typer.
- If you need the package-level ownership split versus USR and OPAL, use [cluster ownership boundary](docs/concepts/ownership-boundary.md) and [cluster semantic surface](docs/concepts/semantic-surface.md).

## Task routes

- Run the first `fit -> umap -> analyze` pass for a chosen feature definition: [exploratory clustering workflow](docs/workflows/exploratory-clustering.md).
- Reuse OPAL outputs for exploratory plots or summaries: [cluster CLI contracts](docs/reference/cli-contracts.md#opal-join-contract).
- Inspect workspaces, presets, and results layout before editing configs: [cluster CLI contracts](docs/reference/cli-contracts.md#workspaces-presets-and-results-layout).
- Verify the package after contract, docs, or workspace changes: [cluster verification contract](docs/reference/verification.md).
- Decide whether you should stay in `cluster` or switch to OPAL: [cluster ownership boundary](docs/concepts/ownership-boundary.md).

## Documentation

- [cluster docs by workflow](docs/README.md): comprehensive index for usage flows, maintainers, demos, and reference routes.
- [cluster docs by type](docs/index.md): workflow, concept, and reference split.
- [exploratory clustering workflow](docs/workflows/exploratory-clustering.md): first runnable `fit`, `umap`, `intra-sim`, and `analyze` path.
- [cluster CLI contracts](docs/reference/cli-contracts.md): command surface, workspace/preset layout, public API, OPAL joins, env vars, and troubleshooting.
- [cluster verification contract](docs/reference/verification.md): deterministic preflight/run/verify path for maintainers and refactors.
- [`dnadesign.cluster.api`](api.py): public in-process ad hoc and workspace execution helpers plus run listing.
- [cluster ownership boundary](docs/concepts/ownership-boundary.md): downstream role relative to USR, infer, and OPAL.
- [cluster semantic surface](docs/concepts/semantic-surface.md): package nouns and runtime boundary rules.
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
uv run cluster workspaces where
uv run cluster workspaces init --help
uv run cluster workspaces list
```

## Results and artifacts

- Checked-in reusable inputs live under `workspaces/` and `presets/`.
- Built-in workspace outputs live under `workspaces/<workspace-id>/outputs/cluster/`.
- Ad hoc standalone runs require an explicit `--results-root`.
- `cluster` does not infer runtime state from the current directory.
- Runtime state under the package tree is allowed only for an explicitly selected workspace-owned `workspaces/<workspace-id>/outputs/cluster/` root.
- Attached overlay columns use one contract for both USR datasets and generic files: `cluster__<run>__...`.
- First-class recorded artifacts include fit `fits/<run-slug>/run.json`, UMAP `umap/<run-slug>/umap.json`, analysis `analysis/<run-slug>/analysis.json`, and sweep `sweeps/<run-slug>/sweep.json`.
- For exact results, reuse, and cleanup semantics, see [cluster CLI contracts](docs/reference/cli-contracts.md#results-and-artifacts).
