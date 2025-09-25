# OPAL Transform Strategies

Transforms convert raw measurement tables into the
campaign’s canonical **Y** label **per design**.

## Contract

A transform is a **pure function** registered in `opal.registries.TRANSFORMS`:

```python
def transform_fn(df_tidy: pd.DataFrame, *, params: dict, setpoint_vector: list[float]) -> pd.DataFrame
```

#### Input:

- `df_tidy`: raw measurements (e.g., design or campaign-specific).
- `params`: transform-specific schema and pre-processing controls (from YAML).

#### Output:

- DataFrame with at least columns:

- `id` (design_id)
- `y_vec` (list[float]) — the canonical label for this campaign.

Transforms must fail fast (raise `IngestError`) on schema or invariant violations
(duplicate timepoints, missing states/channels, NaNs/Inf, wrong lengths).