#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cluster-mpl}"

python_files=(
  src/dnadesign/cluster/__init__.py
  src/dnadesign/cluster/contracts.py
  src/dnadesign/cluster/src/analysis/contracts.py
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py
  src/dnadesign/cluster/src/cli/app.py
  src/dnadesign/cluster/src/presets/schema.py
  src/dnadesign/cluster/src/runs/contracts.py
  src/dnadesign/cluster/tests/test_runtime_contracts.py
  src/dnadesign/cluster/tests/test_docs_contract.py
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py
)

uv run pytest -q \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
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
