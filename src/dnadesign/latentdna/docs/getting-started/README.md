<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md)

# Run a small local analysis

Create 32 illustrative four-dimensional vectors, reduce them to two dimensions with PCA, plot explained variance and export the reduced matrix. These are synthetic numbers for learning the workflow, not predictions of biological activity. No model download or GPU is needed.

## 1. Install and activate

From a DNADesign checkout:

```bash
uv sync --locked --no-default-groups --extra latentdna
source .venv/bin/activate
```

The remaining commands use the installed `latentdna` executable and can run outside the checkout. Activate an existing LatentDNA environment instead if you already have one.

## 2. Create a workspace and input table

Run this in a location where you want to keep the example:

```bash
mkdir latentdna-demo
cd latentdna-demo
mkdir inputs

python - <<'PYTHON'
from pathlib import Path
import numpy as np
import pandas as pd

# Each row is one sequence representation with a stable example ID.
vectors = np.random.default_rng(17).normal(size=(32, 4)).astype("float32")
rows = pd.DataFrame({
    "id": [f"seq_{i:02d}" for i in range(len(vectors))],
    "embedding": list(vectors),
})
rows.to_parquet(Path("inputs/sequences.parquet"), index=False)
print(len(rows), "rows;", vectors.shape[1], "features per row")
PYTHON
```

Expected output:

```text
32 rows; 4 features per row
```

## 3. Declare the view and export

Save the following configuration with the shell command below. A **source** describes the input table; a **view** selects its vector column. The export declaration names the reduced view that the next step will create.

```bash
cat > config.yaml <<'YAML'
schema_version: latentdna.workspace.v1
workspace:
  id: local_example
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  sequences:
    kind: parquet
    path: inputs/sequences.parquet
    record_key: id
    subject_key: id
views:
  embeddings:
    source: sequences
    vector:
      kind: column
      name: embedding
    coordinate_space_id: example_features
exports:
  reduced_features:
    row_basis: example_pca_view
    blocks:
      - kind: reduced_view
        block_id: pca
        source: example_pca_view
        feature_prefix: pca
YAML
```

Both keys use `id` because this example has one representation per sequence. For paired contexts or repeated observations, distinguish row identity from the biological subject using the [source](../reference/source-contract.md) and [alignment](../reference/alignment-contract.md) references.

## 4. Validate, materialize and reduce

```bash
# Check configuration and source columns before creating a matrix.
latentdna validate workspace --workspace . --deep

# Copy the declared vectors into a workspace-owned matrix and row ledger.
latentdna view materialize embeddings --workspace . --quiet
latentdna view stats embeddings --workspace .

# Fit PCA and save both the fitted reducer and the two-dimensional rows.
latentdna view reduce embeddings --workspace . \
  --run-id example_pca --reduced-view-id example_pca_view --dims 2 --quiet

# Plot explained variance from the saved reducer.
latentdna plot render pca_variance --workspace . \
  --kind curve --reducer example_pca --quiet
```

`outputs/views/embeddings/` contains the original 32 × 4 matrix. The fitted PCA state is under `outputs/reducers/example_pca/`, and `outputs/reduced_views/example_pca_view/` contains the 32 × 2 reduced matrix. Open `outputs/plots/pca_variance/plot.svg` or `plot.png` to inspect the variance plot.

## 5. Export and inspect the result

```bash
# Build the export declared in config.yaml.
latentdna export matrix reduced_features --workspace . --quiet

# Check matrix dimensions and the matching row ledger.
python - <<'PYTHON'
from pathlib import Path
import numpy as np
import pandas as pd

bundle = Path("outputs/exports/reduced_features")
print(np.load(bundle / "matrix.npy").shape)
print(len(pd.read_parquet(bundle / "rows.parquet")), "row IDs")
PYTHON

# Inspect workspace status without writing a snapshot file.
latentdna workspace snapshot --workspace . --json --dry-run
```

The matrix check prints `(32, 2)` and `32 row IDs`. Keep `rows.parquet` and `features.parquet` with `matrix.npy`; they identify its rows and columns. Each artifact also has a `manifest.json` recording how it was produced.

## Continue with your data

Replace the source path and vector column with your own precomputed representations, or use [Infer sidecars or a matrix bundle](../reference/source-contract.md). Choose an [analysis workflow](../workflows/README.md) before adding comparisons. For reruns after changing inputs, see [checking and rebuilding artifacts](../operations/README.md).
