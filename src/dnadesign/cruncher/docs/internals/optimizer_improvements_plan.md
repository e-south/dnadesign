# Optimizer improvements plan

## Contents
- [Current behavior](#current-behavior)
- [Open improvement candidates](#open-improvement-candidates)
- [Performance notes](#performance-notes)
- [UX notes](#ux-notes)

## Current behavior

- Parallel tempering is the only optimizer and is treated as the default kernel.
- Best-hit selection is deterministic (score, leftmost start, plus strand).
- Elite selection is TFBS-core MMR with dsDNA canonicalization under bidirectional scoring.
- Early stopping uses plateau detection and optional unique-success gating.
- Run manifests record the effective PT ladder and resolved softmin settings for auditability.

## Open improvement candidates

- Reduce evaluation cost for single-base moves (incremental rescoring rather than full rescans).
- Consolidate PT diagnostics into a single summary block with cold-chain-only ESS conventions.
- Add a compact per-run “optimizer health” summary that mirrors the CLI warnings used by auto-opt.

## Performance notes

- The scorer already caches DP tables for logp; consider reusing per-TF scan buffers across moves.
- If MMR selection becomes a bottleneck on large pools, consider precomputing core strings once per candidate.

## UX notes

- Clarify chain vs draw indexing in reports to match the trace layout shown by ArviZ.
- Add a short glossary for PT ladder terms (beta, temperature, swap acceptance) in the CLI reference.
