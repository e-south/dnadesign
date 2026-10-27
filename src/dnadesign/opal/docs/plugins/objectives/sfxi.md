---
id: opal-objective-sfxi-v1
title: SFXI objective
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-14
---

## setpoint_fidelity_x_intensity `sfxi`

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

`sfxi_v1` converts vec8 model outputs into score and uncertainty channels using
the equations below. It is an objective plugin, not a selector. A selection
view may pair its channels with `top_n` or, when the uncertainty contract is
satisfied, `expected_improvement`.

### At a glance (plugin contract)

- Objective plugin: `sfxi_v1`
- Input shape: `y_pred` vec8 = `[logic(4), log2_intensity(4)]` in state order `[00,10,01,11]`
- Primary score channel: `sfxi` (maximize)
- Additional score channels: `logic_fidelity`, `effect_scaled`
- Uncertainty channel key (when available): `sfxi` (standard deviation of the scalar score)
- Uncertainty method: `delta` (gradient delta-method); it is the only accepted value and the default when omitted or null
- Uncertainty with missing model std: when `y_pred_std` is absent, no uncertainty channel is emitted; an explicit method is still validated
- Uncertainty with model std: required `y_pred_std` entries must be strictly `> 0`; non-positive required entries fail fast
- Uncertainty output contract: emitted scalar uncertainty is finite and non-negative
- Uncertainty clipping contract: derivatives are zero outside the interior of the logic and effect clips; a fully saturated score therefore emits zero local uncertainty instead of spurious variation
- EI boundary: `expected_improvement` separately requires uncertainty to be strictly positive and fails fast on a zero emitted by a saturated score
- Std semantics: `y_pred_std` is interpreted as a standard deviation in objective units and may exceed `1`
- Scaling source: denominator is computed from current-round observed labels and persisted in run metadata
- Audit API: `dnadesign.opal.score_vec8_with_denom` recomputes scores from a
  persisted positive denominator; it is for ledger audits and deterministic
  reranking, not new-round denominator fitting
- Strictness: run fails if current-round labels are fewer than `scaling.min_n`
- Selection wiring:
  - Top-N: `selection.params.score_ref = "sfxi"`
  - EI: `selection.params.score_ref = "sfxi"` and `selection.params.uncertainty_ref = "sfxi"`

---

### 1. What the model predicts

The model predicts an **8-vector (Ŷ)** per input sequence (kept as `pred__y_hat_model`). The first four entries describe the **shape** of a two-factor logic response (bounded from 0 to 1). The last four capture **reference-relative fluorescent intensity** per state, stored in **log2 space** for modeling stability.

$$
\underbrace{v_{00}, v_{10}, v_{01}, v_{11}}_{\text{logic in }[0,1]^4}\,\
\underbrace{y^{\star}_{00}, y^{\star}_{10}, y^{\star}_{01}, y^{\star}_{11}}_{\text{log2(abs. fluorescent intensity)}}
$$

* $v \in [0,1]^4$: **observed logic profile** in state order $[00,10,01,11]$.
* $y^\star \in \mathbb{R}^4$: **per-state reference-relative fluorescent intensity** in log2 space.

### 1.1 From experimental data → 8-vector

The label starts from fluorescent readouts for each state $i$:
$Y^{\mathrm{RFU}}_i$ (YFP/OD600) and $C^{\mathrm{RFU}}_i$ (CFP/OD600).

Each experiment includes a **reference strain** with constitutive YFP. Its
per-state signal anchors instrument and time variation while preserving
reference-relative fluorescent intensity. Let

$$
A_{\mathrm{experiment},i} := \mathrm{mean}\{\text{references' } Y^{\mathrm{RFU}}_i\}
$$

### (a) Logic (fluorophore ratio → log2 → per-design min–max)

A dual fluorescent reporter helps [separate intrinsic from extrinsic noise](https://pmc.ncbi.nlm.nih.gov/articles/PMC3141918/).
The YFP/CFP ratio controls shared cellular and measurement variation, log2
makes fold changes symmetric, and per-design min-max maps the four states to a
common $[0,1]$ range.

$$
r_i = \frac{Y^{\mathrm{RFU}}_i + \varepsilon}{C^{\mathrm{RFU}}_i + \varepsilon}
\qquad
u_i = \log_2(r_i)
$$

$$
u_{\min}=\min_i u_i\quad u_{\max}=\max_i u_i
$$

$$
v_i = \frac{u_i - u_{\min}}{(u_{\max}-u_{\min})+\eta} \in [0,1]
$$

If $u_{\max}\approx u_{\min}$ (flat logic), set $v_i=\tfrac{1}{4}$ for all $i$ and **warn**. ($\varepsilon,\eta>0$ are small stabilizers and are recorded in metadata.)

### (b) Reference-relative fluorescent intensity (reference-normalized → log2)

Reference-relative intensity carries the **effect size** or **scale** signal
while raw RFUs can drift by experiment and time. Division by the experiment's
reference strain produces a unitless quantity. Log2 storage limits leverage
from unusually bright samples; objective evaluation returns it to linear
space.

$$
y^{\mathrm{linear}}_i = \frac{Y^{\mathrm{RFU}}_i}{A_{\mathrm{experiment},i} + \alpha}
\quad\text{(unitless, reference-relative fluorescence)}
$$

$$
y^\star_i = \log_2\ \big(y^{\mathrm{linear}}_i + \delta\big)
$$

**8-vector label (stored under `y_column_name`):**

$$
Y = [v_{00}, v_{10}, v_{01}, v_{11}, y^\star_{00}, y^\star_{10}, y^\star_{01}, y^\star_{11}]
$$

### 1.1c Time reduction is an upstream label contract

`sfxi_v1` scores a vec8 and does not choose an assay time or integration
window. The producing Reader record must declare how each per-state value was
reduced from its time series. Changing that reduction creates a different
label version; it must not silently overwrite an existing observed-label
ledger.

A snapshot and a fixed-window summary can both preserve the SFXI premise when
they keep the two axes separate:

- logic is reduced from `YFP/CFP`, then transformed to the four-state `v`;
- reference-relative fluorescence is reduced from `YFP/OD600` against the same-state reference
  strain over the same time support, then stored as `y*`.

For an AUC-based summary, use one prespecified window, interpolate its
boundaries, and divide area by window duration before comparing experiments.
Raw AUC values from different durations are not commensurate. The reference
anchor must use the same window and state as the sample. Snapshot and window
summaries remain distinct provenance-bearing label artifacts until a
prospective assay comparison promotes one contract.

The Reader response-window handoff
`[r00, r10, r01, r11, b00, b10, b01, b11]` is not that SFXI window label.
Its `r` values are reduced log2 response values rather than declared `v`
values, and its `b` values belong to the response-metric fluorescence contract.
Do not normalize or rename that handoff into an SFXI vec8.

### 1.2 Modeling note (median–IQR robust scaling):

Random forests can fit the mixed vec8 target, but low-sample per-state
log-intensities can differ in variance and influence split decisions unevenly.
An affine, monotonic, reversible median-IQR transform places the four
intensity targets on comparable scales.

* Fit-time transform (applied to all training samples, per state): compute campaign-cumulative training median and IQR for each intensity target $y^\star_i$, then

$$
\tilde{y}^\star_i
= \frac{\,y^\star_i - \mathrm{median}_{\text{train}}(y^\star_i)\,}
         {\max\ \big(\mathrm{IQR}_{\text{train}}(y^\star_i),\varepsilon\big)}
$$

This centers typical values near 0 and makes one unit approximately one IQR.
The bounded logic block $v$ is not scaled.

### 1.3 Inference-time inversion (undo scaling → undo log)

After model fitting, predictions return to linear, reference-relative
fluorescence. Reversing the campaign-wide affine transform and log transform
preserves ordering and restores interpretable magnitudes.

$$
\widehat{y}^\star_i = \widehat{\tilde{y}}^\star_i \,\mathrm{IQR}_{\text{train}}(y^\star_i)+\mathrm{median}_{\text{train}}(y^\star_i) \qquad \widehat{y}^{\mathrm{linear}}_i = \max\!\bigl(0,\; 2^{\,\widehat{y}^\star_i} - \delta\bigr)
$$

**Note on $\max(0,\cdot)$ and $\delta$:** The $\delta>0$ term is the same small offset used when taking $\log_2(y^{\mathrm{linear}}+\delta)$ to avoid $\log(0)$. Subtracting $\delta$ undoes that offset; the outer $\max(0,\cdot)$ guards against tiny negative values from numerical round-off, ensuring the recovered $\widehat{y}^{\mathrm{lin}}_i$ remains non-negative.

---

### 2. Inputs to the objective (selection time)

* **Predictions:** $\widehat{v}\in[0,1]^4$ and $\widehat{y}^{\star}\in\mathbb{R}^{4}$ (log2 intensity block from the vec8 model output).
* **Setpoint** (i.e., preference): $p\in[0,1]^4$. This can be binary setpoints (e.g., AND: $[0,0,0,1]$) or nuanced continuous ones (e.g., $[0.3,0.4,0.7,0.2]$).

Each candidate is scored with $p$, $\widehat{v}$, and $\widehat{y}^{\star}$; OPAL converts $\widehat{y}^{\star}$ to linear intensity internally before computing effect terms.

---

### 3. Logic fidelity

Root-mean-square error compares the predicted logic vector
$\widehat{v}\in[0,1]^4$ with setpoint $p\in[0,1]^4$. Normalization maps the
error to a similarity in $[0,1]$: 1 is an exact match and 0 is the farthest
possible corner for that setpoint.

**“Worst-case” error for this setpoint.** Inside the unit 4-cube $[0,1]^4$, the farthest point from $p$ is a **corner**. For each state $i$, choose whichever of $\{0,1\}$ is farther from $p_i$
(e.g., if $p_i=0.6$, distance to 0 is $0.6$, to 1 is $0.4$ ⇒ pick **0**).
That corner’s Euclidean distance from $p$ is

$$
D \;=\; \sqrt{\sum_{i=1}^{4} \max\!\big(p_i^2,\,(1-p_i)^2\big)}
$$

Examples: $p=[0,0,0,1]\Rightarrow D=2$; $p=[0.5,0.5,0.5,0.5]\Rightarrow D=1$

**Normalized RMSE → similarity.** With four states, $\mathrm{RMSE}(\widehat{v},p)=\tfrac{1}{2}\lVert \widehat{v}-p\rVert_2$ and $\mathrm{RMSE}_{\max}=\tfrac{D}{2}$

Report

$$
F_{\text{logic}}
\;=\; 1 \;-\; \frac{\mathrm{RMSE}(\widehat{v},p)}{\mathrm{RMSE}_{\max}}
\;=\; 1 \;-\; \frac{\lVert \widehat{v}-p\rVert_2}{D}
\qquad
F_{\text{logic}} \leftarrow \mathrm{clip}(F_{\text{logic}},\,0,\,1)
$$

$F_{\text{logic}}=1$ only when $\widehat{v}=p$; $F_{\text{logic}}=0$ at the setpoint’s farthest corner. A value of **0.7** places the joint error at 30% of the worst possible error for this $p$.

For valid $\widehat{v}\in[0,1]^4$, $\lVert \widehat{v}-p\rVert_2\!\le\! D$ so $F_{\text{logic}}\in[0,1]$ already; the clip only guards tiny numerical drift or predictions slightly outside $[0,1]$. Normalizing by $D$ makes scores directly comparable across binary or continuous setpoints.

---

### 4. Evaluating fluorescent intensity in target conditions

The effect term is a setpoint-weighted average of predicted per-state
intensity. It is rescaled with the labeled data for the fitted round to map the
value into $[0,1]$.

**Weights from the setpoint (turn setpoint into state weights).**
Let $p\in[0,1]^4$ and $P=\sum_i p_i$. Define

$$
w_i =
\begin{cases}
\dfrac{p_i}{P}, & P>0\\[4pt]
0, & P=0
\end{cases}
\quad\text{so that } w_i\ge 0 \text{ and } \sum_i w_i=1
$$

This makes a simple average: each state’s contribution is proportional to how much the setpoint values it.

**Raw effect (weighted average of predicted intensities).**
With predicted linear intensities $\widehat{y}^{\mathrm{linear}}_i$

$$
E_{\mathrm{raw}} = \sum_{i=1}^{4} w_i \,\widehat{y}^{\mathrm{linear}}_i
\quad\text{(equivalently } E_{\mathrm{raw}}=\tfrac{p\cdot \widehat{y}^{\mathrm{linear}}}{\max(P,\epsilon)}\text{, with } \epsilon>0 \text{ a small guard)}
$$

Raising intensity where $p_i$ is large **always** increases $E_{\mathrm{raw}}$; intensity where $p_i=0$ does **not**.

If $P=0$ (an “all-OFF” setpoint), define $w=\mathbf{0}$ and set $E_{\mathrm{raw}}=0$; the score is then fully determined by the logic term.


**Round-internal robust scaling.**
$E_{\mathrm{raw}}$ maps to $[0,1]$ using only **the current round's labeled designs**:

* The denominator, **$\mathrm{denom}$**, is the **`scaling.percentile`th percentile** (default 95) of $\{E_{\mathrm{raw}}\}$ recomputed over the round’s labels under the current setpoint $p$, with a small floor $\epsilon>0$:

  $$
  \mathrm{denom} \;=\; \max\!\Big(\text{95th percentile of } \{E^{\text{(round)}}_{\mathrm{raw}}\},\ \epsilon\Big)
  $$
* **Scaled effect:**

  $$
  E_{\mathrm{scaled}} \;=\; \min\!\Big(1,\ \max\!\big(0,\ \tfrac{E_{\mathrm{raw}}}{\mathrm{denom}}\big)\Big)
  $$

Using the **same-round** labeled set makes the scale **self-calibrating** to that experiment/day; the 95th percentile is robust to a few extreme bright wells (they map to ~1). The realized denominator is a **per-run constant** and must be snapshotted in the **round context / objective meta artifact** (referenced by `run_meta`), not duplicated per-ID. As a result, $E_{\mathrm{scaled}}$ is unit-free, bounded, and comparable **within the round**.

**Strictness:** The objective requires at least `scaling.min_n` labeled designs in the **current round**. If there are fewer, the run fails with a clear error (no silent fallbacks). Lower `scaling.min_n` or add labels to proceed.


---

### 5. Final scoring metric

$$
\boxed{\;\text{score} = \big(F_{\text{logic}}\big)^{\beta}\cdot \big(E_{\text{scaled}}\big)^{\gamma}\;}
\qquad \beta=\gamma=1
$$

* $\beta>1$: emphasize logic correctness more strongly.
* $\gamma>1$: emphasize intensity in the desired conditions more strongly.

* **Final product:** Low logic fidelity or low target-state fluorescence depresses the score; both components must be positive for a positive product.

---

### 6. Metric properties

* **Batch robustness.** Logic uses YFP/CFP ratios; intensity uses a **reference-strain anchor**; and selection scales the effect by a **within-round percentile** over labels.
* **Setpoint flexibility.** A changed $p$ defines another selection view over
  the same predicted $\widehat{y}^{\mathrm{linear}}$ when the learning
  lifecycle is otherwise shared.

---

### 6.1 Selection limitations

Canonical `sfxi_v1` remains a reporting baseline. Its product score allows
target-state intensity to compensate for weak setpoint fidelity whenever both
components are nonzero. Its logic term uses per-design min-max scaling, so a
small response span can appear shape-perfect, while `r_logic` remains a
diagnostic rather than part of the score. Its effect term omits brightness in
target-OFF states. The round-fitted intensity denominator also makes scores
comparable within a recorded round, not directly across rounds.

These are properties of the documented equations, not implementation defects.
They make the decomposition useful for reporting but can be unsuitable when a
selection must satisfy response shape, ON-state brightness, and OFF-state
leakage as separate requirements. Use
[Response-Magnitude Feasibility (RMF)](response-magnitude-feasibility.md) for that
binary-mask, non-compensatory objective. It requires raw state responses and
explicitly calibrated constraints; it is not a reinterpretation of vec8 and
is a distinct objective with its own input contract.

---

### 7. Concrete examples (state order $[00,10,01,11]$)

### (a) AND-like setpoint

$p=[0,0,0,1]\Rightarrow w=[0,0,0,1]$.

* Candidate A: $\widehat{y}^{\mathrm{linear}}=[0.1,0.2,0.3,1.2]$ → $E_{\text{raw}}=1.2$ (all credit from $A{+}B$).
* Candidate B: $\widehat{y}^{\mathrm{linear}}=[0.8,0.9,0.7,0.2]$ → $E_{\text{raw}}=0.2$ (intensity in the wrong conditions doesn’t help).
  If B’s $\widehat{v}$ is also far from $p$, $F_{\text{logic}}$ is small → score stays low.

### (b) Nuanced setpoint

$p=[0.3,0.4,0.7,0.2]\Rightarrow P=1.6,\; w\approx[0.1875,0.25,0.4375,0.125]$.

* Candidate C: $\widehat{y}^{\mathrm{linear}}=[0.2,0.8,0.2,0.8]$ → $E_{\text{raw}}\approx 0.425$.
* Candidate D: $\widehat{y}^{\mathrm{linear}}=[0.1,0.5,1.1,0.4]$ → $E_{\text{raw}}\approx 0.675$.
  Similar total signal; D wins because it is intense in the **high-$p$** state (01).

### (c) “All-OFF” setpoint

$p=[0,0,0,0]\Rightarrow w=0\Rightarrow E_{\text{raw}}=0$.
Only proximity of $\widehat{v}$ to $p$ (being OFF everywhere) is rewarded.

---

### 8. Edge cases and guards
* **Tiny CFP or anchor:** add $\varepsilon,\alpha$.
* **Flat logic:** if $u_{\max}\approx u_{\min}$, set $v=\tfrac{1}{4}\mathbf{1}$.
* **Non-finite:** reject at ingestion.
* **Too few labels in round:** objective errors; lower `scaling.min_n` or add labels.
* **Removed analytical approximation:** configuration and runtime validation reject `uncertainty_method=analytical`. That approximation did not implement the clipped nonlinear score and could report uncertainty where the score was locally constant.
* **Uncertainty validity:** required model std entries must be positive where they affect the score. Computed scalar uncertainty must be finite and non-negative; zero is valid objective output for a locally constant clipped score.
* **Clip-aware delta derivatives:** logic gradients are active only for raw predictions strictly inside `(0, 1)`, and effect gradients are active only while `E_raw` is strictly inside `(0, denominator)`. The method does not propagate local variance through a saturated clip.
* **Delta exact-setpoint cusp:** delta-method uncertainty fails fast when candidates land exactly on the logic setpoint (`dist=0`) and this would otherwise produce zero uncertainty from a non-differentiable logic branch.
* **Fractional-exponent derivatives:** for `0 < logic_exponent_beta < 1` or `0 < intensity_exponent_gamma < 1`, delta-method derivatives are singular at base `0`; OPAL fails fast if `F_logic <= 0` (beta case) or `E_scaled <= 0` (gamma case).

---

### 9. Emissions

All outputs are written to **ledger sinks** under `outputs/ledger/`:

- `outputs/ledger/predictions/` → per‑ID `run_pred` rows (one per candidate)
- `outputs/ledger/runs.parquet` → per‑run `run_meta` row (one per run)
- `outputs/ledger/labels.parquet` → per‑label `label` rows (one per ingest event)

Selection channels and diagnostics are distinct:

- channel refs (`score_ref`, `uncertainty_ref`) resolve only objective channels
- `obj__*` columns are analysis diagnostics and are not selectable channel refs

**Per-ID predictions (`run_pred` rows)**

- `pred__y_hat_model`: one shared objective-space phenotype prediction
- `pred__score_channels`: namespaced SFXI channels such as `<view>/sfxi`
- `pred__selection_views`: view-local score, uncertainty, diagnostics, rank,
  and selected flag

Public analysis readers project a chosen view to `view__selection_score`,
`view__rank_competition`, `view__is_selected`, and `view__diagnostics`.

**Per‑run metadata (`run_meta` row)**

- `objective__defs_json`: view-indexed SFXI parameters and emitted channels
- `selection_views__defs_json`: view-indexed refs, selector parameters,
  summary statistics, and tie behavior
- `training__y_ops` — Y‑ops applied at fit time (inverted before objective)

`round_ctx.json` also records:

- `objective/sfxi_v1/denom_value`
- `objective/sfxi_v1/denom_percentile`
- full plugin contract audit trail under `core/contracts/...`

**Recomputable at runtime (not persisted per‑ID)**

- $E_{\text{raw}}$ from `pred__y_hat_model[4:8]` using the setpoint‑derived weights
- $F_{\text{logic}}$ from `pred__y_hat_model[0:4]` and the setpoint
- $D$ (worst‑case distance) from the setpoint
- $E_{\text{scaled}}$ via `objective__denom_value` (from `run_meta` or `round_ctx.json`)

---

### Appendix

**Why an 8-vector and not a 4-vector?**

The 8‑vector is minimal and justified for the above stated objective. It cleanly separates:

* **Shape (right):** ratio‑based, effect size‑invariant logic.
* **Scale (bright):** reference-anchored intensity.

The calculation retains eight values per design:

* **4 logic numbers (`v`)** — the **shape** of the response, built from the **YFP/CFP ratio** and min–max scaled across the four states. The ratio cancels extrinsic noise.
* **4 intensity numbers (`y*`)** — the **reference-relative brightness** per state, anchored to a reference strain and stored in log2 for modeling stability.

The ratio-based logic (`v`) cannot be recovered from YFP intensities alone.
Min-max scaling YFP across states retains state-specific capacity shifts that
the YFP/CFP ratio is intended to remove. Reducing the representation to four
intensities omits information required by SFXI and therefore computes a
different score.

---
