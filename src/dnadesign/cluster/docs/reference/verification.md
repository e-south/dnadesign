## cluster verification contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Use this page when you are changing `cluster` code, docs, or checked-in jobs/presets and want the smallest deterministic verification path.

### Start here

1. Run the fast verification script first.
2. Use the manual breakdown only when you need to widen or isolate one failure.
3. If you changed method math or broader dataflow behavior, widen beyond this package-local surface after the fast path is green.

### Fast verify path

```bash
bash src/dnadesign/cluster/scripts/verify_cluster_contracts.sh
```

### What the fast path checks

- runtime contracts and run-artifact typing
- docs progressive-disclosure routes and reference links
- public import boundary plus CLI bootstrap side effects
- targeted `ruff` and `py_compile` checks
- primary CLI help surfaces for `fit`, `umap`, `analyze`, and `sweep`

### Manual breakdown

#### Preflight

```bash
git status --short src/dnadesign/cluster
uv run cluster --help
```

#### Run

```bash
uv run pytest -q \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py
```

#### Verify

```bash
uv run ruff check \
  src/dnadesign/cluster/__init__.py \
  src/dnadesign/cluster/contracts.py \
  src/dnadesign/cluster/src/analysis/contracts.py \
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py \
  src/dnadesign/cluster/src/cli/app.py \
  src/dnadesign/cluster/src/presets/schema.py \
  src/dnadesign/cluster/src/runs/contracts.py \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py

uv run ruff format --check \
  src/dnadesign/cluster/__init__.py \
  src/dnadesign/cluster/contracts.py \
  src/dnadesign/cluster/src/analysis/contracts.py \
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py \
  src/dnadesign/cluster/src/cli/app.py \
  src/dnadesign/cluster/src/presets/schema.py \
  src/dnadesign/cluster/src/runs/contracts.py \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py

uv run python -m py_compile \
  src/dnadesign/cluster/__init__.py \
  src/dnadesign/cluster/contracts.py \
  src/dnadesign/cluster/src/analysis/contracts.py \
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py \
  src/dnadesign/cluster/src/cli/app.py \
  src/dnadesign/cluster/src/presets/schema.py \
  src/dnadesign/cluster/src/runs/contracts.py \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py
```

### When to widen

- If you changed clustering method implementation, rerun the broader `cluster` tests beyond the contract set above.
- If you changed cross-tool docs or upstream/downstream handoffs, rerun the repository docs checks as well.
- If you changed runtime artifact schemas, inspect representative `run.json`, `umap.json`, or `analysis/analysis.json` outputs in a writable results root after the fast path.

### Related docs

- [cluster CLI contracts](cli-contracts.md)
- [cluster semantic surface](../concepts/semantic-surface.md)
- [exploratory clustering workflow](../workflows/exploratory-clustering.md)
