## Promoter Characterization Feature Matrix

**Type:** runbook
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** one or more USR-backed promoter datasets plus optional construct-expanded context datasets
**Exit artifact:** infer-annotated USR feature matrix ready for cluster or OPAL

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Use this runbook when promoter candidates come from multiple USR-backed sources and downstream consumers should see one infer-annotated feature matrix with explicit provenance.

This is the authoritative cross-tool runbook for:

- upstream promoter sources such as DenseGen anchors plus wildtype or manually imported promoters,
- optional construct expansion into larger contexts such as 1 kb windows,
- explicit infer job matrices across model lanes and output planes, and
- downstream handoff into clustering, OPAL, or other tools that consume one chosen `infer__...` column.

This runbook owns the data-plane workflow through infer write-back. It does not own cluster internals or the OPAL active-learning loop; those remain downstream tool docs.

### Boundary decisions

- USR owns the durable source of truth and cross-tool merge/carry semantics.
- `densegen` and manual import paths only need to produce valid USR datasets plus any overlays that should survive consolidation.
- `construct` is optional contextualization. Use it only when a larger template-backed sequence context is part of the representation study.
- `infer` owns the feature matrix write-back through explicit `infer__<model>__<job>__<out>` columns.
- `cluster` and `opal` consume one chosen infer-derived `X` column; they do not decide how the upstream feature matrix is assembled.
- No hidden combinatorial wrapper is introduced here. Context choice, model lane, and output plane stay explicit as dataset ids, infer configs, and job ids.

### When to use this path

- anchor-only sequences from DenseGen or curated wildtype promoters should be compared in the same study,
- one branch should stay at the original sequence length while another branch is expanded through construct,
- infer should annotate those branches with log likelihoods, pooled logits, or pooled embeddings,
- downstream exploration should use one consolidated dataset contract before Leiden clustering or active learning starts.

### Design stance

- Keep source assembly, optional construct expansion, and infer write-back as separate reversible steps.
- Treat anchor-only and construct-expanded contexts as explicit dataset planes. Do not hide context expansion inside infer configs.
- Keep one infer config per model lane (`evo2_7b`, `evo2_20b`, and so on). Vary context and output plane through explicit job ids inside that lane.
- Start with a tracer-bullet matrix:
  - one anchor-only dataset,
  - one optional construct-expanded dataset,
  - one smaller model lane such as `evo2_7b`,
  - one scalar output (`ll`) plus one pooled intermediate embedding (`mid`).
- Add `final` embeddings, pooled logits, or larger model lanes only after the first write-back path is green.

## Ordered procedure

### 1) Choose the source datasets and the downstream feature dataset

Assume the upstream source datasets already exist in one USR root. Use the shared multi-source runbook first if they still need explicit merge/carry setup.

```bash
export USR_ROOT=/abs/path/to/usr_root # Reuse one explicit USR root across merge, construct, infer, cluster, and OPAL.
export PRIMARY_INPUT_DATASET="promoter_sources_control" # Wildtype, manual, or seeded promoter records.
export EXTRA_INPUT_DATASET="promoter_sources_densegen" # DenseGen-derived anchor records that should join the same study.
export FEATURE_DATASET="promoter_feature_matrix_demo" # One semantic dataset id for downstream infer-derived feature columns.
```

Two context planes are supported here:

- anchor-only plane: reuse the merged upstream dataset directly as `FEATURE_DATASET`
- construct-expanded plane: materialize a downstream construct dataset first, then treat that construct dataset as `FEATURE_DATASET`

### 2) Assemble the upstream promoter sources explicitly

If source rows already live in multiple USR datasets, merge them explicitly before any construct or infer step:

```bash
uv run usr --root "$USR_ROOT" validate "$PRIMARY_INPUT_DATASET" --strict # Validate the primary source dataset before mutation.
uv run usr --root "$USR_ROOT" validate "$EXTRA_INPUT_DATASET" --strict # Validate the additional source dataset before merge.
# Merge the DenseGen or auxiliary promoter source into the primary source dataset.
uv run usr --root "$USR_ROOT" maintenance merge \
  --dest "$PRIMARY_INPUT_DATASET" \
  --src "$EXTRA_INPUT_DATASET" \
  --union-columns \
  --if-duplicate error \
  --carry-namespace usr_label
# Inspect the merged source dataset with carried label overlays.
uv run usr --root "$USR_ROOT" head "$PRIMARY_INPUT_DATASET" -n 10 \
  --columns id,usr_label__primary,usr_label__aliases
```

If anchor-only sequences are the intended infer input, set:

```bash
export FEATURE_DATASET="$PRIMARY_INPUT_DATASET" # Reuse the merged anchor-only dataset as the infer target plane.
```

### 3) Optionally materialize larger construct-backed contexts

When the representation study needs 1 kb windows, plasmid contexts, or other template-backed sequences, materialize those explicitly with construct first. Reuse the shared construct-backed source-of-truth runbooks instead of duplicating their unique steps here:

- [Multi-source source-of-truth assembly](multi-source-source-of-truth-assembly.md): use when the merged source dataset should be realized through construct into one downstream dataset.
- [Construct -> USR -> Infer source-of-truth demo](construct-infer-source-of-truth-demo.md): use when one construct-backed downstream dataset is already the intended canonical handoff.

After construct materializes the expanded context dataset, set:

```bash
export FEATURE_DATASET="multi_source_construct_truth_demo" # Reuse the construct-backed dataset as the infer target plane.
```

### 4) Define the infer job matrix explicitly

Use one infer config per model lane. Inside that config, keep the job ids explicit so downstream tools can choose feature columns without ambiguity.

Recommended job-id pattern:

- `<context>_<model>_<signal>`
- examples:
  - `anchor_7b_ll`
  - `anchor_7b_emb_mid`
  - `window_7b_emb_mid`
  - `window_20b_emb_final`

Recommended first-pass matrix:

| Context plane | Model lane | Output plane | Why start here |
| --- | --- | --- | --- |
| anchor-only | `evo2_7b` | `ll_mean` | cheapest scalar signal and easiest write-back smoke |
| anchor-only | `evo2_7b` | `emb_mid` | stable pooled intermediate embedding baseline |
| construct-expanded | `evo2_7b` | `emb_mid` | compare larger-context representation without changing model lane |
| construct-expanded | `evo2_20b` | `emb_mid` | optional higher-capacity comparison once the 7B path is green |

Example infer config fragment for one model lane:

```yaml
model: # Configure one explicit Evo2 model lane per infer config.
  id: evo2_7b # Select the Evo2 7B lane for the first tracer bullet.
  device: cpu # Keep the local smoke path on CPU before widening to GPU lanes.
  precision: fp32 # Use an explicit precision for repeatable local validation.
  batch_size: 2 # Keep the first smoke batch small and reversible.

jobs: # Keep context and output variation explicit as job ids inside the model lane.
  - id: anchor_7b_ll # Anchor-only scalar likelihood baseline.
    ingest: # Read directly from the anchor-only USR dataset.
      source: usr # Use the USR ingest surface.
      root: /abs/path/to/usr_root # Resolve the canonical USR root explicitly.
      dataset: promoter_sources_control # Consume the merged anchor-only promoter dataset.
      field: sequence # Read the sequence field from USR rows.
    outputs: # Emit one explicit scalar output plane.
      - id: ll_mean # Stable output id for the pooled likelihood signal.
        fn: evo2.log_likelihood # Call the Evo2 likelihood extractor.
        format: float # Persist the result as a scalar column.

  - id: anchor_7b_emb_mid # Anchor-only pooled intermediate embedding baseline.
    ingest: # Read the same anchor-only USR dataset.
      source: usr # Use the USR ingest surface.
      root: /abs/path/to/usr_root # Resolve the canonical USR root explicitly.
      dataset: promoter_sources_control # Consume the merged anchor-only promoter dataset.
      field: sequence # Read the sequence field from USR rows.
    outputs: # Emit one pooled embedding output plane.
      - id: emb_mid # Stable output id for the pooled intermediate embedding.
        fn: evo2.embedding # Call the Evo2 embedding extractor.
        params: # Configure the semantic layer alias and pooling policy.
          layer: mid # Use the common intermediate-layer alias.
          pool: { method: mean, dim: 1 } # Mean-pool across the sequence axis.
        format: list # Persist the result as a vector column.

  - id: window_7b_emb_mid # Construct-expanded pooled intermediate embedding baseline.
    ingest: # Read from the larger construct-backed context dataset.
      source: usr # Use the USR ingest surface.
      root: /abs/path/to/usr_root # Resolve the canonical USR root explicitly.
      dataset: multi_source_construct_truth_demo # Consume the construct-expanded context dataset.
      field: sequence # Read the sequence field from USR rows.
    outputs: # Emit one pooled embedding output plane.
      - id: emb_mid # Stable output id for the pooled intermediate embedding.
        fn: evo2.embedding # Call the Evo2 embedding extractor.
        params: # Configure the semantic layer alias and pooling policy.
          layer: mid # Use the common intermediate-layer alias.
          pool: { method: mean, dim: 1 } # Mean-pool across the sequence axis.
        format: list # Persist the result as a vector column.
```

Notes:

- use `layer: mid` as the stable semantic intermediate-layer alias for pooled Evo2 embeddings
- use `layer: final` only as an explicit comparison lane
- keep sequence-window choice outside infer by pointing different jobs at the anchor-only or construct-expanded dataset plane
- copy the config and change `model.id` when you need a second model lane such as `evo2_20b`

### 5) Validate, register, and dry-run the infer matrix

```bash
export INFER_CONFIG_7B=/abs/path/to/infer.promoter-matrix.7b.yaml # Keep one explicit config path for the 7B lane.
uv run infer validate config --config "$INFER_CONFIG_7B" # Validate config schema and runtime contracts.
uv run infer validate usr-registry --config "$INFER_CONFIG_7B" # Render the exact infer namespace registration required for write-back.
uv run infer run --config "$INFER_CONFIG_7B" --dry-run # Preflight the USR dataset plus infer job matrix without model execution.
```

If you add a second model lane, repeat the same preflight for the second config:

```bash
export INFER_CONFIG_20B=/abs/path/to/infer.promoter-matrix.20b.yaml # Keep one explicit config path for the optional 20B lane.
uv run infer validate config --config "$INFER_CONFIG_20B" # Validate the second model-lane config before any execution.
uv run infer validate usr-registry --config "$INFER_CONFIG_20B" # Render namespace registration requirements for the second lane.
uv run infer run --config "$INFER_CONFIG_20B" --dry-run # Preflight the second model lane without model execution.
```

### 6) Execute the selected matrix slices and inspect the resulting feature columns

```bash
# Execute the selected 7B infer matrix lane against the chosen feature dataset.
uv run infer run --config "$INFER_CONFIG_7B"
# Inspect the resulting infer-derived feature columns on the feature dataset.
uv run usr --root "$USR_ROOT" head "$FEATURE_DATASET" -n 5 \
  --columns id,usr_label__primary,infer__evo2_7b__anchor_7b_ll__ll_mean,infer__evo2_7b__anchor_7b_emb_mid__emb_mid
```

Expected outcome:

- the feature dataset remains a valid USR dataset
- construct lineage remains present when the construct-expanded plane is used
- infer columns are queryable through explicit `infer__...` names
- downstream tools can now pick one concrete `infer__...` column as `X`

### 7) Branch into downstream consumers

#### Cluster branch

Use `cluster` when the immediate goal is exploratory structure, Leiden clustering, UMAP visualization, or OPAL-joined diagnostics later.

```bash
# Fit one Leiden clustering run against the chosen infer-derived feature column.
uv run cluster fit \
  --dataset "$FEATURE_DATASET" \
  --x-col infer__evo2_7b__anchor_7b_emb_mid__emb_mid \
  --name promoter_matrix_ldn_v1 \
  --write \
  --allow-overwrite
# Render one UMAP view against the same infer-derived feature column.
uv run cluster umap \
  --dataset "$FEATURE_DATASET" \
  --name promoter_matrix_ldn_v1 \
  --x-col infer__evo2_7b__anchor_7b_emb_mid__emb_mid \
  --attach-coords \
  --write \
  --allow-overwrite
```

Continue with [cluster exploratory clustering workflow](../../../cluster/docs/workflows/exploratory-clustering.md) for preset, hue, and OPAL-join details.

#### OPAL branch

Use OPAL when the feature dataset is ready and the next step is explicit label/train/select rounds. The downstream owner is:

- [USR dataset with infer-derived X -> OPAL active learning](../../../opal/docs/workflows/usr-infer-x-active-learning.md)

That workflow starts after the feature dataset already has the chosen `infer__...` column:

```bash
uv run opal validate -c "$OPAL_WORKDIR/configs/campaign.yaml" # Validate the USR-backed OPAL campaign against the chosen infer-derived X column.
uv run opal run -c "$OPAL_WORKDIR/configs/campaign.yaml" --labels-as-of 0 # Train, score, and select once the first observed labels are available.
```

Continue with the OPAL-owned workflow for the full label-ingest and round-loop procedure.

## Verification checklist

- `usr maintenance merge ... --carry-namespace usr_label` succeeds when multiple source datasets are present
- the chosen feature dataset validates under `uv run usr --root "$USR_ROOT" validate "$FEATURE_DATASET" --strict`
- `infer validate config`, `infer validate usr-registry`, and `infer run --dry-run` pass for every model-lane config
- `usr head "$FEATURE_DATASET"` shows explicit `infer__...` columns for the selected job ids
- `cluster fit` or the OPAL workflow can consume one chosen `infer__...` column without additional hidden repointing

## Related docs

- Root docs router: [../../../../../docs/README.md](../../../../../docs/README.md)
- USR operations index: [README.md](README.md)
- USR workflow map: [workflow-map.md](workflow-map.md)
- Multi-source upstream assembly: [multi-source-source-of-truth-assembly.md](multi-source-source-of-truth-assembly.md)
- Construct-backed source-of-truth handoff: [construct-infer-source-of-truth-demo.md](construct-infer-source-of-truth-demo.md)
- Infer docs router: [../../../infer/docs/README.md](../../../infer/docs/README.md)
- Cluster exploratory workflow surface: [../../../cluster/docs/workflows/exploratory-clustering.md](../../../cluster/docs/workflows/exploratory-clustering.md)
- OPAL downstream workflow: [../../../opal/docs/workflows/usr-infer-x-active-learning.md](../../../opal/docs/workflows/usr-infer-x-active-learning.md)
