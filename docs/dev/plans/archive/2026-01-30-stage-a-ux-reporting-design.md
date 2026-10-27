## Stage-A Progress + Recap UX Simplification


### Contents
- [Context](#context)
- [Goals](#goals)
- [Non-goals](#non-goals)
- [Decisions](#decisions)
- [Live Table (Default)](#live-table-default)
- [Recap Table (Default)](#recap-table-default)
- [Data Contracts](#data-contracts)
- [Logging Behavior](#logging-behavior)
- [Testing Plan](#testing-plan)
- [Docs Updates](#docs-updates)

### Context
Stage-A live progress and recap tables are information-dense but confusing in practice.
Non-monotonic percentages (from changing denominators), redundant columns, and verbose
warnings between progress and recap increase cognitive load and hide key signals.

### Goals
- Make the live mining table monotonic and low-noise.
- Keep the recap audit-grade but compact by default.
- Remove ambiguous/low-signal columns (e.g., gen %, has_hit) from default views.
- Enforce assertive contracts (no silent fallbacks for missing fields).
- Preserve existing sampling semantics and metrics; only presentation changes.

### Non-goals
- Change Stage-A sampling or selection behavior.
- Remove audit metrics from the manifest.
- Reduce the ability to debug (verbose mode will keep full detail).

### Decisions
- Live table removes `gen %` entirely. It shows only `generated/limit` (fixed)
  to avoid non-monotonic progress.
- Live table drops `has_hit`, `eligible_raw`, and `tier target` columns.
- Recap table becomes compact by default, with a `--verbose` mode for full detail.
- Progress/recap rows are constructed from typed dataclasses with required fields.
  No `get(...)` fallbacks in default views; missing fields are hard errors.
- Stream (non-screen) mode prints phase transition lines only; no batch spam.

### Live Table (Default)
Columns:
- motif
- phase
- generated/limit
- eligible_unique/target
- tier yield (0.1/1/9)
- batch
- elapsed
- rate

Notes:
- `generated/limit` uses `max_candidates` (or time budget if time-bound), so it is
  monotonic and stable.
- `phase` explicitly shows `mining`, `fimo`, `postprocess` to explain stalls.

### Recap Table (Default)
Columns:
- TF
- generated
- eligible_unique
- retained
- tier fill
- selection
- k(pool/target)
- div(pairwise)
- delta div(pairwise)
- overlap
- delta score (median)
- score(min/med/max)
- len(min/med/max)

Verbose mode adds:
- has_hit
- eligible_raw
- set_swaps
- delta score (p10)
- any additional diagnostics currently stored in the manifest

### Data Contracts
Introduce typed row objects:
- StageAProgressRow
- StageARecapRow

Construction validates required fields and raises clear errors on missing data.
This prevents silent regressions in CLI/plotting.

### Logging Behavior
- Screen mode: Live progress updates in place.
- Stream mode: One line per phase transition; no batch spam.
- Tier target warnings emitted once after progress completes; recap value only
  shows the tier fraction (no "unmet (need ...)" in table cells).

### Testing Plan
- Unit tests for compact vs verbose recap columns.
- Unit tests for live table headers (no gen % column).
- Stream mode test to ensure phase transition output is single-line.
- Contract tests to assert missing fields raise errors.

### Docs Updates
- Update sampling and outputs docs to reflect compact/verbose modes and the
  new column definitions.
- Document the stable denominator for `generated/limit` and rationale for
  removing `gen %`.
