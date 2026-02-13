# Cruncher v3 Single-Path Numeric Schema Design

## Contents
- [Intent and Invariants](#intent-and-invariants)
- [Schema Surface](#schema-surface)
- [Runtime Mapping](#runtime-mapping)
- [Plan of Action](#plan-of-action)
- [Related Docs](#related-docs)

## Intent and Invariants
Cruncher is an optimization system for fixed-length dsDNA sequence design that jointly satisfies multiple PWMs while returning a diverse elite set. The core intent is to maximize per-TF consensus-like scores, then diversify the resulting sequences across the candidate pool. Diversity is defined across sequences, not between TF windows inside a single sequence. The MCMC kernel remains unchanged; changes are confined to configuration, orchestration, and selection. The system is strict by design: no silent fallbacks, no hidden auto-disable behavior, and explicit validation of all invariants. Reproducibility is mandatory: fixed inputs, fixed RNG seeds, and explicit manifests of resolved settings.

The invariants are: (1) sequence length is fixed and must be at least the widest PWM; (2) bidirectional scoring implies canonicalization, and uniqueness is enforced on canonical sequences; (3) MMR selection operates on TFBS-core windows oriented to each PWM and compares only like-to-like TF cores across sequences; (4) selection and reporting must be deterministic with explicit tie-break rules; and (5) all configuration is research-grade, numeric, and explicit, avoiding categorical presets. The schema should reflect user intent directly, not internal architecture boundaries. Compute is treated as a single numeric budget measured in sweeps, with a deterministic adaptation phase.

## Schema Surface
The schema exposes a single usage path with numeric units. Compute is controlled by `sample.compute.total_sweeps` and `sample.compute.adapt_sweep_frac`. There is no separate `tune/draws` input or tuning mode switch; adaptation is always enabled during the initial fraction of sweeps. Objective and selection parameters remain explicit in their natural sections. The schema surface prioritizes clarity and interpretability over configurability.

Proposed keys:

```yaml
sample:
  seed: 42
  sequence_length: 30

  compute:
    total_sweeps: 4000
    adapt_sweep_frac: 0.25

  elites:
    k: 10
    min_per_tf_norm: 0.70
    mmr_alpha: 0.85
```

Notes:
- `sequence_length` is the explicit invariant; validation rejects lengths shorter than the widest PWM.
- `total_sweeps` is the only compute budget exposed to users.
- `adapt_sweep_frac` is numeric and explicit, enabling deterministic adaptation windows.
- `min_per_tf_norm` is the explicit consensus-like gate, preserving research-grade thresholds.
- `mmr_alpha` is the explicit relevance-vs-diversity tradeoff.
- PT internals (beta ladders, swap cadence, move ranges) are removed from user config and fixed internally, but recorded in manifests for auditability.

## Runtime Mapping
At runtime, `adapt_sweeps = ceil(total_sweeps * adapt_sweep_frac)`. PT ladder adaptation runs for `adapt_sweeps`, then the ladder is frozen for the remaining sweeps. There are no pilot runs or auto-opt grids. Selection happens once at the end of the run via MMR on TFBS-core windows, using canonical sequences when bidirectional scoring is enabled. MMR distances are computed between same-TF core windows across sequences; there is no cross-TF comparison within a sequence. Canonicalization and uniqueness enforcement happen before MMR selection to ensure deterministic diversity outcomes. All resolved internal values (adapt_sweeps, final ladder, swap acceptance, scoring parameters) are persisted in `config_used.yaml` and the run manifest.

Reports and diagnostics use the same terminology: “total_sweeps”, “adapt_sweeps”, and “mmr_alpha”. Any warnings or errors must be explicit and deterministic. The reporting surface should never imply inference correctness; diagnostics should be framed as optimizer health signals.

## Plan of Action
1. Update configuration schema and validators: rename `sample.length` to `sample.sequence_length`, add `sample.compute.total_sweeps` and `sample.compute.adapt_sweep_frac`, remove `tune/draws` and all auto-opt/pilot keys, flatten and rename elite controls to `sample.elites.min_per_tf_norm` and `sample.elites.mmr_alpha`, and require `elites.k >= 1`.
2. Update runtime wiring: compute `adapt_sweeps`, run ladder adaptation during that window, remove pilot paths, and enforce canonicalization/uniqueness under bidirectional scoring.
3. Update manifests and reporting to use the new terminology and record derived values.
4. Update docs and demos to reflect the new schema and remove deprecated knobs; add TOCs and clear cross-links for navigation.
5. Update tests to reject removed keys, validate fixed-length invariants, and exercise the new schema end-to-end with a CLI smoke test.

## Related Docs
- `src/dnadesign/cruncher/docs/reference/config.md`
- `src/dnadesign/cruncher/docs/guides/sampling_and_analysis.md`
- `src/dnadesign/cruncher/README.md`
