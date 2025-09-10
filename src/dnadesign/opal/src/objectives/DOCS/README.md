# OPAL Objectives — Docs

> **Scope:** This folder documents OPAL *objectives* only, which convert a model’s predicted outputs into a **scalar score** used by selection.

### Inventory

| name                   | expects                                                                | returns                                | brief features                                                                  |
| ---------------------- | ---------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- |
| `setpoint_fidelity_x_intensity_p95_v1` | `y_pred: (N,5)` where `v[:4]∈[0,1]` (logic), `v[4]≥0` (effect, linear) | `score: (N,)` (+ optional diagnostics) | setpoint-aware logic fidelity × percentile-scaled effect; gate-free, continuous |


### **Rules** (succinct):

- `y_pred` semantics are **entry-specific** (documented per entry).
- **No ranking here** (selection handles ordering/ties).
- Names are registered in `opal.registries.objectives` and autoloaded.

---

@e-south