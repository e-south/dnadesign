# OPAL Model and Selection Strategy Matrix

Use this page to pick a model/selection combination that matches your campaign goal.

## Terms

- **Model**: predicts `y_hat`; may also emit predictive standard deviation.
- **Objective**: converts `y_hat` into named score channels and optional uncertainty channels.
- **Selection**: ranks candidates using `selection.params.score_ref` and optional `uncertainty_ref`.

In OPAL, "optimizer choice" is the model + objective + selection trio.

## Built-in combinations

| Flow | Model | Selection | Use when | Required refs |
| --- | --- | --- | --- | --- |
| Baseline deterministic | `random_forest` | `top_n` | Fast baseline ranking with no acquisition term | `score_ref` |
| UQ model, deterministic selection | `gaussian_process` | `top_n` | You want GP predictions but deterministic rank-by-score | `score_ref` |
| UQ acquisition | `gaussian_process` | `expected_improvement` | You want exploration/exploitation behavior | `score_ref`, `uncertainty_ref` |

## Selection semantics

- `top_n`: ranks strictly by selected score channel.
- `expected_improvement`: ranks by acquisition score computed from selected score + selected uncertainty standard deviation channel.

`expected_improvement` fails fast if `uncertainty_ref` is missing, non-finite, negative, or all-zero.

## Wiring checklist

1. Select one or more objective plugins under `objectives`.
2. Choose `selection.params.score_ref` as `<objective_name>/<score_channel_name>`.
3. Set `selection.params.objective_mode` to match that score channel mode.
4. For EI, set `selection.params.uncertainty_ref` to a standard-deviation channel.
5. Set `selection.params.tie_handling` and `top_k`.

## Demo mappings

- RF + SFXI + top_n: `docs/guides/demos/rf-sfxi-topn.md`
- GP + SFXI + top_n: `docs/guides/demos/gp-sfxi-topn.md`
- GP + SFXI + expected_improvement: `docs/guides/demos/gp-sfxi-ei.md`
