# DenseGen Stage-B Sampler Refactor Design

## Intent

Reduce Stage-B sampling coupling by splitting the monolithic orchestration logic into focused, testable components while preserving current behavior. The refactor must be assertive (fail fast), modular, and avoid fallbacks.

## Problem Statement

`orchestrator.py` currently mixes Stage-B library construction, sampling loop control, diagnostics logging, and run-level persistence. This makes it harder to change Stage-B behavior without touching unrelated responsibilities, and increases the cognitive load for maintenance.

## Approaches Considered

1. **Minimal extraction**: Move the Stage-B loop into a single helper class and keep all logic together.
   - Pros: smallest diff, easy to review.
   - Cons: still conflates feasibility checks, sampling loop control, and diagnostics.

2. **Moderate modularization (recommended)**: Split into 2–3 focused components with explicit contracts.
   - Pros: improves cohesion, supports targeted tests, preserves current behavior.
   - Cons: moderate refactor size, requires careful wiring.

3. **Full rewrite**: Redesign Stage-B as a state machine.
   - Pros: clean architecture.
   - Cons: too risky and not YAGNI for current goals.

## Proposed Architecture (Option 2)

### Components

1. **LibraryBuilder** (`core/pipeline/stage_b_library_builder.py`)
   - Responsibility: construct the next Stage-B library and validate feasibility.
   - Inputs: plan item, pool strategy, pool artifacts, config.
   - Output: `LibraryContext` (TFBS list, TF list, IDs, fixed-element metrics, feasibility breakdown).
   - Invariants:
     - When `groups` exist, `required_regulators_selected` must be present.
     - `sequence_length` must satisfy fixed-element and per-TF minimums.
     - Raise on infeasible constraints.

2. **StageBSampler** (`core/pipeline/stage_b_sampler.py`)
   - Responsibility: run the sampling loop and manage resampling logic.
   - Inputs: `LibraryContext`, runtime policy, optimizer/generator deps, diagnostics.
   - Output: `SamplingResult` (generated count, resamples, duplicates, failure counts, leaderboard snapshot).
   - Invariants:
     - Respect `max_consecutive_failures` and `iterative_max_libraries`.
     - Fail fast if `pool_strategy` does not allow resampling.

3. **SamplingDiagnostics** (`core/pipeline/sampling_diagnostics.py`)
   - Responsibility: failure tracking and leaderboard summaries.
   - Called by StageBSampler; no file I/O or persistent state of its own.

### Orchestrator Role

`orchestrator.py` remains an assembly layer: config resolution, run-level persistence, event emission, and sink flushing. It wires the components and remains the only place that performs run-level I/O. Any failed I/O should abort the run (no silent fallbacks).

## Data Flow

1. Orchestrator resolves config, plan, and pool dependencies.
2. `LibraryBuilder.build_next()` returns a validated `LibraryContext`.
3. `StageBSampler.run()` executes the sampling loop with `SamplingDiagnostics` updates.
4. Orchestrator persists outputs (attempts, solutions, metrics), emits events, and updates run state.

## Error Handling

- **LibraryBuilder** raises on infeasible constraints or missing required regulator metadata.
- **StageBSampler** raises on disallowed resampling or limit breaches.
- **Orchestrator** treats all output/event errors as fatal to keep state consistent.

## Testing Strategy

### Unit tests
- `LibraryBuilder` validates feasibility and invariants.
- `StageBSampler` enforces resample limits and produces expected counters with a stub optimizer.
- `SamplingDiagnostics` is already isolated and covered via sampler tests.

### Integration tests
- Re-run existing tests that touch `_process_plan_for_source` to confirm behavior parity.

## Rollout

Implement in small commits:
1. Add `LibraryBuilder` + tests.
2. Add `StageBSampler` + tests.
3. Rewire orchestrator and confirm tests pass.

## Non-Goals

- No schema changes.
- No behavior changes to sampling heuristics or scoring logic.
- No fallbacks or backward compatibility shims.
