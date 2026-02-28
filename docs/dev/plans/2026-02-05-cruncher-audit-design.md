## Cruncher Audit Design (2026-02-05)


### Contents
- [Intent](#intent)
- [Scope](#scope)
- [Audit Method](#audit-method)
- [Deliverables](#deliverables)
- [Change Boundaries](#change-boundaries)
- [Invariants to Enforce](#invariants-to-enforce)
- [Verification (expected)](#verification-expected)

### Intent
Audit Cruncher end-to-end with a pragmatic, no-fallback posture: high cohesion/low coupling, explicit invariants, and docs that match real behavior. Focus on PT-only sampling, fixed-length constraints, and MMR canonical elites. Prioritize clear, didactic documentation and workspace config alignment over broad test expansion.

### Scope
In:
- `src/dnadesign/cruncher/` code, docs, tests, and workspaces
- Primary CLI flow (two-TF demo sample → analyze)
- Config schema simplification and invariant enforcement
- Profiling hot paths (cProfile)
- Documentation alignment (guides, reference, demos)

Out:
- Changes to core MCMC acceptance kernel
- Large refactors outside Cruncher
- Broad test bloat beyond essential invariants

### Audit Method
1. **Run Cruncher tests** and collect failures/warnings.
2. **Run primary CLI flow** using `workspaces/demo_basics_two_tf` to verify artifacts, plots, and reporting.
3. **Read docs** (guides, reference, demo) and check for mismatches against runtime behavior.
4. **Profile demo run** to identify hot paths and opportunities for safe, explicit reuse.
5. **File-organizer lens**: identify oversized modules and misplaced responsibilities, but do not move files without approval.

### Deliverables
- **Findings**: brief prioritized list of bugs, footguns, and doc/behavior mismatches.
- **Fixes**: targeted, minimal changes aligned with no-fallback behavior.
- **Docs**: update `sampling_and_analysis.md`, `config.md`, and demo docs to match behavior.
- **Workspaces**: align all Cruncher configs with the simplified schema. The `three_tfs` workspace remains densegen-oriented but must be schema-correct and modern.
- **Verification commands**: explicit commands to reproduce tests and CLI flow.

### Change Boundaries
- Tests only where they protect critical invariants (e.g., fixed length vs max PWM width, CLI smoke path). Avoid test bloat beyond essential invariant protection.
- Plots should remain minimal and meaningful. Remove stale or misleading plots; add only if clearly actionable.
- Any caching or reuse must be explicit, deterministic, and documented (no hidden fallbacks).

### Invariants to Enforce
- `sample.init.length >= max_pwm_width` is a hard requirement.
- PT-only execution across CLI/config.
- MMR is canonical when bidirectional scoring is enabled.
- Auto-opt uses PT pilots and fixed-length sampling.

### Verification (expected)
- `uv run pytest -q`
- `uv run cruncher sample -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml`
- `uv run cruncher analyze -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml`
