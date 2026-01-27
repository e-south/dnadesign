---
title: Prom60 SFXI Diagnostics + Uncertainty (Developer Notes)
---

# Prom60 SFXI diagnostics + uncertainty (developer notes)

## Purpose & operator workflow

These diagnostics extend the prom60 dashboard to make **active learning choices** more transparent:

1) **Verify logic shape**: inspect factorial effects and nearest-gate class to confirm predicted logic matches the desired gate family.
2) **Understand failure modes**: decompose distance-to-setpoint to see which states drive mismatch.
3) **Check support / extrapolation**: quantify distance to labeled logic (and UMAP-space neighbors) before selecting risky candidates.
4) **Assess uncertainty**: identify high-score but high-uncertainty candidates that warrant cautious exploration.
5) **Tune setpoint safely**: sweep alternative setpoints without retraining to assess sensitivity and intensity scaling.

## Mathematical definitions

**State order is strict**: `[00, 10, 01, 11]` for all logic vectors.

### Factorial effects (from logic vector `v`)

Given `v = (v00, v10, v01, v11)` in `[0,1]^4`:

```
A_effect = ((v10 + v11) - (v00 + v01)) / 2
B_effect = ((v01 + v11) - (v00 + v10)) / 2
AB_interaction = ((v11 + v00) - (v10 + v01)) / 2
```

### Nearest gate (truth-table library)

Let `G` be the 16 binary truth tables (vectors in `{0,1}^4` in state order). For each candidate:

```
nearest_gate_class = argmin_g ||v_hat - g||_2
nearest_gate_dist  = min_g ||v_hat - g||_2
```

### Distance to labeled logic (support)

Use **observed** label logic (`v_obs`) for support:

```
dist_to_labeled_logic = min_j ||v_hat(candidate) - v_obs(label_j)||_2
```

### Distance to labeled X (UMAP space)

If UMAP coordinates exist:

```
dist_to_labeled_x = min_j ||x_umap(candidate) - x_umap(label_j)||_2
```

This is a **visual support diagnostic** only (no objective coupling).

### SFXI objective components (reused from `sfxi_v1`)

See `docs/setpoint_fidelity_x_intensity.md` for full derivation. The dashboard reuses the same math:

```
F_logic = 1 - ||v_hat - p||_2 / D(p)
E_raw   = sum_i w_i * y_hat_linear_i
denom   = percentile_p(E_raw on current-round labels)
E_scaled = clip(E_raw / denom, 0, 1)
score   = (F_logic^beta) * (E_scaled^gamma)
```

`w_i = p_i / sum(p)` if `sum(p) > 0`, else `w = 0` (all‑OFF setpoint → intensity ignored).

## Required inputs & invariants (strict)

**Core columns** (predictions):
- `pred_y_hat` (vec8, objective space, length 8; dashboard view)
- `pred__y_hat_model` (vec8, objective space, length 8; ledger/CLI)
- `pred_score` or `opal__view__score`
- `pred_logic_fidelity`, `pred_effect_raw`, `pred_effect_scaled` (or overlay equivalents)

**Labels (observed)**:
- `y_obs` (vec8, length 8) for logic support and denom computation

**Run/objective params**:
- setpoint vector `p` in `[0,1]^4`
- `beta`, `gamma`, `delta`
- scaling config `{percentile, min_n, eps}`

**Invariants**:
- State order is always `[00,10,01,11]`.
- Vectors must be finite and correct length; missing/invalid inputs raise actionable errors.
- denom is computed **from current‑round labels** (same as `sfxi_v1`).

## Plot semantics & interpretation

### A) Factorial‑effects map
- **x:** `A_effect`, **y:** `B_effect`, **color:** `AB_interaction`
- Optional size: `effect_scaled` (default)
- Overlay markers: labeled, selected, top‑k (if available)
- Sampling + rasterization are explicit; annotate the number of points shown.

### B) Setpoint decomposition
Per‑state residuals: `|v_hat - p|` (2×2 heatmap aligned to state order).
Per‑state intensity contribution: `w_i * y_hat_linear_i`.

If `sum(p) == 0`, intensity panel is **all zeros** and explicitly labeled
“all‑OFF setpoint ⇒ intensity ignored”.

### C) Setpoint sweep (objective landscape)
Library = **16 truth tables + current setpoint**. For each setpoint:
- median `F_logic` on labels‑as‑of
- **top‑k mean** `F_logic` (k configurable)
- fraction `F_logic > tau`
- `denom_used` from current‑round labels
- clip fractions for `E_scaled` (labels; optional pool sample)
  - pool clip fractions must use the **label-derived denom** (objective‑consistent scaling)

### D) Logic support diagnostics
Scatter: x=`dist_to_labeled_logic`, y=`score` (or `F_logic`), color=`E_scaled` or `F_logic`.

### E) X‑space support (UMAP)
Scatter in UMAP space; overlay labeled + selected. Color by score or `nearest_gate_class`.
This is a support visualization only (no objective math).

### F) Uncertainty views
Scatter `uncertainty` vs `score` (color by `F_logic` or `E_scaled`).
Optional scatter `uncertainty` vs `dist_to_labeled_logic`.

**Contract**: uncertainty supports `kind="score"` and `kind="y_hat"` (default `score`).
If unsupported, the UI errors explicitly.

### G) Intensity scaling diagnostics
Per‑setpoint:
- `denom_used` vs setpoint
- clip fractions vs setpoint
- distribution of `E_raw` (labels, optional pool sample)

Denom definition must be explicit (e.g., “95th percentile of `E_raw` on current‑round labels”).

## Architecture plan (extensible + DRY)

**Objective math (single source of truth)**
`src/dnadesign/opal/src/objectives/sfxi_math.py`
- `logic_fidelity`, `weights_from_setpoint`, `recover_linear_intensity`
- `effect_raw`, `effect_scaled`, `denom_from_labels`

**Diagnostics math (no UI, no plotting)**
`src/dnadesign/opal/src/analysis/sfxi/`
- `state_order.py` (single source of truth for `[00,10,01,11]`)
- `factorial_effects.py`
- `gates.py` (16 truth tables + nearest gate)
- `support.py` (dist to labeled logic / UMAP)
- `intensity_scaling.py`
- `setpoint_sweep.py`
- `uncertainty.py` (model‑agnostic contract + RF adapter)

**Chart builders (matplotlib, no marimo)**
`src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_*.py`
- Each exports `make_<plot>_figure(...) -> matplotlib.figure.Figure`

**Notebook UI only orchestrates**
`notebooks/prom60_eda.py`:
- validate inputs
- compute derived metrics (once, cached)
- call chart builders
- layout panels (diagnostics column is penultimate)

## TODOs (explicit)

1) **Intensity proxy for factorial size**: default `effect_scaled` (chosen).
2) **Setpoint sweep library**: 16 truth tables + current setpoint (chosen).
3) **Denom source of truth**: confirm **current‑round labels** (aligned with `sfxi_v1`).
4) **Uncertainty kind**: support `score` + `y_hat` (default `score`).
5) **Derived metrics placement**: in‑memory only; export toggle TBD.
6) **UMAP support distance**: `dist_to_labeled_x` uses UMAP coords only.

## Testing checklist

- [ ] Factorial effects math (exact expected values)
- [ ] Nearest gate assignment (truth table self‑match, near‑match)
- [ ] dist_to_labeled_logic (min L2 distance)
- [ ] Setpoint sweep denom + clip fractions
- [ ] RF uncertainty shape + non‑negativity (`score` + `y_hat`)
- [ ] prom60 notebook smoke test still passes
