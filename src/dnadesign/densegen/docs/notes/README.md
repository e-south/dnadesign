# DenseGen Notes (rolling, compact)

These are **persistent but transient** notes: short, actionable findings that
inform future refactors. They are not a spec.

## 2026-01-14

- PWM sampling now accepts MEME, JASPAR (PFM), and CSV matrices; all map to the
  same BindingSiteRecord model with `densegen__input_mode: pwm_sampled`.
- Parquet metadata must remain native list/struct columns (no JSON strings);
  regenerate old datasets instead of migrating.

## 2026-01-15

- Dense-arrays regulator constraints are now used directly; pipeline still validates for
  `approximate` strategy to keep behavior consistent.
- Regulator labels are carried explicitly (no reliance on parsing `tf:tfbs` strings).
