---
id: opal-demo-scalar-fixture
title: OPAL synthetic scalar fixture
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
surface: opal_demo_fixture
---

## Synthetic scalar fixture

This fixture contains 96 generated DNA-like sequences, twelve synthetic input
features per row, and scalar labels for 32 rows. It contains no experimental
records, source paths, study identifiers, or biological claims.

Rebuild it from the repository root:

```bash
uv run python -m dnadesign.devtools.fixtures.opal_scalar_demo
```

The OPAL demo matrix copies these files into temporary campaign workspaces.
They are not examples of a scientific objective.
