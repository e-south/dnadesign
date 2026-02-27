## OPAL — Optimization with Active Learning

![OPAL banner](assets/opal-banner.svg)

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for biological sequence design.

1. Train a configured model on your chosen representation **X** and observed labels **Y**.
2. Predict outcomes for candidate records with valid **X**.
3. Evaluate configured objectives into named score and uncertainty channels.
4. Select candidates using the configured selection strategy and explicit channel refs.
5. Append runtime events to ledger sinks under `outputs/`.
6. Persist per-round artifacts (model, selection CSV, round context, objective metadata, logs) for auditability.

The pipeline is plugin-driven: swap data transforms (X/Y), models, objectives, and selection in `configs/campaign.yaml` without editing core runtime code.

### Documentation

1. [Open the documentation index](./docs/index.md).
2. [Run an RF + SFXI + top_n workflow](./docs/workflows/rf-sfxi-topn.md).
3. [Run a GP + SFXI + top_n workflow](./docs/workflows/gp-sfxi-topn.md).
4. [Run a GP + SFXI + expected_improvement workflow](./docs/workflows/gp-sfxi-ei.md).
5. [Use the CLI reference for command contracts and flags](./docs/reference/cli.md).
6. [Read Gaussian Process behavior and math](./docs/plugins/model-gaussian-process.md).
7. [Read Expected Improvement behavior and math](./docs/plugins/selection-expected-improvement.md).
8. [Read SFXI behavior and math](./docs/plugins/objective-sfxi.md).
