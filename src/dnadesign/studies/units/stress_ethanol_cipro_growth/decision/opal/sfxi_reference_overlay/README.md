---
id: stress-ethanol-cipro-sfxi-reference-overlay
title: SFXI reference overlay
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-08-01
---

# SFXI reference overlay

This package owns the stress study's historical SFXI overlay recipe. It keeps
three independent contracts joined without transferring ownership:

- Reader publishes digest-addressed `four_state_vector/vector` records under
  `logic.four_state_vector.v1`; it does not assign them an objective.
- This study asks OPAL to score those eight coordinates as SFXI through
  `dnadesign.opal.api.sfxi`.
- USR publishes the resulting additive overlay through atomic
  `Dataset.create_overlay`.

The default command is a read-only preview:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.sfxi_reference_overlay \
  --reader-root ../reader
```

Add `--write` only when the target dataset does not already contain an
`sfxi_ref` overlay. The command preserves the existing collection, campaign,
metric, and historical provenance wire identifiers. That historical provenance
value is evidence already present in the published overlay, not a current
Reader contract name. The recipe does not import Reader Python code or infer
replicate relationships. `reader_records.json` pins the canonical
record contract, revision, content digest, configuration digest, and selected
design identities. Any drift fails before scoring.
