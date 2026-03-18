#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cluster-mpl}"

python_files=(
  src/dnadesign/cluster/api.py
  src/dnadesign/cluster/__init__.py
  src/dnadesign/cluster/contracts.py
  src/dnadesign/cluster/src/analysis/contracts.py
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py
  src/dnadesign/cluster/src/cli/app.py
  src/dnadesign/cluster/src/cli/commands.py
  src/dnadesign/cluster/src/cli/commands_analysis.py
  src/dnadesign/cluster/src/cli/commands_fit.py
  src/dnadesign/cluster/src/cli/commands_table.py
  src/dnadesign/cluster/src/cli/commands_umap.py
  src/dnadesign/cluster/src/cli/resolution.py
  src/dnadesign/cluster/src/cli/subapps.py
  src/dnadesign/cluster/src/cli/umap_resolution.py
  src/dnadesign/cluster/src/execution.py
  src/dnadesign/cluster/src/execution_analysis.py
  src/dnadesign/cluster/src/execution_analysis_support.py
  src/dnadesign/cluster/src/execution_fit.py
  src/dnadesign/cluster/src/execution_fit_support.py
  src/dnadesign/cluster/src/execution_sweep.py
  src/dnadesign/cluster/src/execution_support.py
  src/dnadesign/cluster/src/execution_table.py
  src/dnadesign/cluster/src/execution_umap.py
  src/dnadesign/cluster/src/io/parquet_attach.py
  src/dnadesign/cluster/src/layout.py
  src/dnadesign/cluster/src/methods/__init__.py
  src/dnadesign/cluster/src/methods/kmeans.py
  src/dnadesign/cluster/src/methods/params.py
  src/dnadesign/cluster/src/methods/registry.py
  src/dnadesign/cluster/src/presets/runtime.py
  src/dnadesign/cluster/src/presets/schema.py
  src/dnadesign/cluster/src/runs/contracts.py
  src/dnadesign/cluster/src/runs/index.py
  src/dnadesign/cluster/src/runs/index_store.py
  src/dnadesign/cluster/src/umap/contracts.py
  src/dnadesign/cluster/src/umap/frame.py
  src/dnadesign/cluster/src/umap/hues.py
  src/dnadesign/cluster/src/umap/overlays.py
  src/dnadesign/cluster/src/umap/plot.py
  src/dnadesign/cluster/src/umap/requests.py
  src/dnadesign/cluster/src/util/meta.py
  src/dnadesign/cluster/src/workspaces/__init__.py
  src/dnadesign/cluster/src/workspaces/contracts.py
  src/dnadesign/cluster/src/workspaces/errors.py
  src/dnadesign/cluster/src/workspaces/loader.py
  src/dnadesign/cluster/src/workspaces/paths.py
  src/dnadesign/cluster/src/workspaces/schema.py
  src/dnadesign/cluster/tests/test_runtime_contracts.py
  src/dnadesign/cluster/tests/test_source_tree_contracts.py
  src/dnadesign/cluster/tests/test_docs_contract.py
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py
)

uv run pytest -q \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_source_tree_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py

uv run ruff check "${python_files[@]}"
uv run ruff format --check "${python_files[@]}"
uv run python -m py_compile "${python_files[@]}"

uv run cluster --help >/dev/null
uv run cluster fit --help >/dev/null
uv run cluster umap --help >/dev/null
uv run cluster analyze --help >/dev/null
uv run cluster sweep --help >/dev/null
uv run cluster workspaces where >/dev/null
uv run cluster workspaces list >/dev/null
uv run cluster runs list --workspace promoter_clusters_v1 >/dev/null

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/cluster-verify.XXXXXX")"
trap 'rm -rf "$tmp_root"' EXIT

cat >"$tmp_root/records.csv" <<'EOF'
id,x1,x2,source
a,0.0,0.1,A
b,0.0,0.2,A
c,10.0,10.2,B
d,10.1,10.3,B
e,20.0,20.1,C
f,20.2,20.3,C
EOF

uv run cluster workspaces init --id verify_ws --root "$tmp_root" >/dev/null

cat >"$tmp_root/verify_ws/config.yaml" <<'EOF'
schema_version: 1
input:
  file: "../records.csv"

fit:
  name: "verify_ws"
  key_col: "id"
  x_cols: "x1,x2"
  method: "leiden"
  method_params:
    neighbors: 2
    resolution: 0.2
  write: true
  allow_overwrite: true
  inplace: true

umap:
  name: "verify_ws"
  key_col: "id"
  x_cols: "x1,x2"
  neighbors: 2
  min_dist: 0.1
  metric: "euclidean"
  random_state: 42
  attach_coords: true
  write: true
  allow_overwrite: true
  inplace: true
  plot:
    enabled: false
    dims: [4, 4]
    alpha: 0.7
    size: 10.0

analyze:
  cluster_col: "cluster__verify_ws"
  group_by: "source"
  composition: true
  plots: false
EOF

uv run cluster fit --workspace "$tmp_root/verify_ws" >/dev/null
uv run cluster umap --workspace "$tmp_root/verify_ws" >/dev/null
uv run cluster analyze --workspace "$tmp_root/verify_ws" >/dev/null
uv run cluster sweep \
  --workspace "$tmp_root/verify_ws" \
  --method leiden \
  --method-param neighbors=2 \
  --res-min 0.1 \
  --res-max 0.1 \
  --step 0.1 \
  --seeds 1 >/dev/null

uv run cluster runs list --workspace "$tmp_root/verify_ws" >/dev/null

cat >"$tmp_root/api_smoke.py" <<'EOF'
from pathlib import Path

from dnadesign import cluster

workspace = Path(__import__("os").environ["CLUSTER_VERIFY_WORKSPACE"])
records = cluster.list_workspace_runs(workspace)
assert {"fit", "umap", "analysis", "sweep"} <= set(records["kind"].tolist())

api_root = workspace.parent / "verify_api_ws"
cluster.init_workspace(workspace_id="verify_api_ws", root=workspace.parent)
(api_root / "config.yaml").write_text(
    """
schema_version: 1
input:
  file: "../records.csv"

fit:
  name: "verify_api_ws"
  key_col: "id"
  x_cols: "x1,x2"
  method: "leiden"
  method_params:
    neighbors: 2
    resolution: 0.2
  write: true
  allow_overwrite: true
  inplace: true

umap:
  name: "verify_api_ws"
  key_col: "id"
  x_cols: "x1,x2"
  neighbors: 2
  min_dist: 0.1
  metric: "euclidean"
  random_state: 42
  attach_coords: true
  write: true
  allow_overwrite: true
  inplace: true
  plot:
    enabled: false
    dims: [4, 4]
    alpha: 0.7
    size: 10.0

analyze:
  cluster_col: "cluster__verify_api_ws"
  group_by: "source"
  composition: true
  plots: false
""".strip()
    + "\n",
    encoding="utf-8",
)

cluster.run_fit_workspace(api_root)
cluster.run_umap_workspace(api_root)
cluster.run_analyze_workspace(api_root)
cluster.run_sweep_workspace(
    api_root,
    overrides={
        "method": "leiden",
        "method_params": {"neighbors": 2},
        "res_min": 0.1,
        "res_max": 0.1,
        "step": 0.1,
        "seeds": "1",
    },
)
api_runs = cluster.list_workspace_runs(api_root)
assert {"fit", "umap", "analysis", "sweep"} <= set(api_runs["kind"].tolist())

adhoc_root = workspace.parent / "verify_api_results"
cluster.run_fit(
    results_root=adhoc_root,
    file=workspace.parent / "records.csv",
    name="verify_api_adhoc",
    key_col="id",
    x_cols=("x1", "x2"),
    method="leiden",
    method_params={"neighbors": 2, "resolution": 0.2},
    write=True,
    allow_overwrite=True,
    inplace=True,
)
cluster.run_umap(
    results_root=adhoc_root,
    file=workspace.parent / "records.csv",
    name="verify_api_adhoc",
    key_col="id",
    x_cols=("x1", "x2"),
    neighbors=2,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
    attach_coords=True,
    write=True,
    allow_overwrite=True,
    inplace=True,
    render_plots=False,
    plot={"dims": [4, 4], "alpha": 0.7, "size": 10.0},
)
cluster.run_analyze(
    results_root=adhoc_root,
    file=workspace.parent / "records.csv",
    cluster_col="cluster__verify_api_adhoc",
    group_by="source",
    composition=True,
    plots=False,
)
cluster.run_sweep(
    results_root=adhoc_root,
    file=workspace.parent / "records.csv",
    key_col="id",
    x_cols=("x1", "x2"),
    method="leiden",
    method_params={"neighbors": 2},
    res_min=0.1,
    res_max=0.1,
    step=0.1,
    seeds=(1,),
)
adhoc_runs = cluster.list_runs(adhoc_root)
assert {"fit", "umap", "analysis", "sweep"} <= set(adhoc_runs["kind"].tolist())
EOF

CLUSTER_VERIFY_WORKSPACE="$tmp_root/verify_ws" uv run python "$tmp_root/api_smoke.py" >/dev/null

test -f "$tmp_root/verify_ws/outputs/cluster/index.parquet"
test -f "$tmp_root/records.csv"
find "$tmp_root/verify_ws/outputs/cluster/verify_ws/fits" -name 'run.json' -print -quit | grep -q 'run.json'
find "$tmp_root/verify_ws/outputs/cluster/verify_ws/fits" -name 'labels.parquet' -print -quit | grep -q 'labels.parquet'
find "$tmp_root/verify_ws/outputs/cluster/verify_ws/umap" -name 'coords.parquet' -print -quit | grep -q 'coords.parquet'
find "$tmp_root/verify_ws/outputs/cluster/verify_ws/analysis" -name 'analysis.json' -print -quit | grep -q 'analysis.json'
find "$tmp_root/verify_ws/outputs/cluster" -path '*/sweeps/*/sweep.json' -print -quit | grep -q 'sweep.json'
find "$tmp_root/verify_api_results/verify_api_adhoc/fits" -name 'run.json' -print -quit | grep -q 'run.json'
