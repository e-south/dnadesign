# DenseGen Stage-A audit (2026-01-30)

- Scope: Stage-A PWM sampling behavior and diagnostics for demo_sampling_baseline.
- Branch: densegen/cruncher-refine.
- Workspace: src/dnadesign/densegen/workspaces/demo_sampling_baseline.
- Command: uv run dense stage-a build-pool --fresh (config auto-detected).
- Inputs: pwm_artifact_set (LexA + CpxR).

## Stage-A intent and rationale (code-aligned)
- Stage-A builds TF-specific TFBS pools from PWM artifacts using FIMO log-odds scoring (forward strand only, thresh=1.0).
- Eligibility requires at least one hit and best_hit_score > 0; sequences are deduplicated by core (matched_sequence) before ranking.
- Ranking is score-first with deterministic tie-breaks (lexicographic sequence).
- Diagnostic tiers (0.1%, 1%, 9%, rest) and tier-targeted mining provide budget control and early warnings.
- MMR selection trades score vs diversity on tfbs_core with PWM-tolerant weighted Hamming distance; shortlist pool built from a tier ladder for headroom.

## Stage-A settings used in demo_sampling_baseline
- n_sites: 500 per motif.
- mining: tier_target (target_tier_fraction=0.001), max_candidates=500000, max_seconds=300, batch_size=5000.
- length: range [15, 20], core embedded in background flanks.
- uniqueness.key: core.
- selection: mmr (alpha=0.9, shortlist_factor=5, shortlist_min=50, tier_widening ladder [0.001, 0.01, 0.09, 1.0]).

## Observed outcomes (from pool_manifest + CLI recap)

### lexA_CTGTATAWAWWHACA
- generated: 500000; candidates_with_hit: 500000; eligible_raw: 499263.
- eligible_unique: 87005 (about 17% of raw; duplication factor about 5.7x).
- retained: 500; tier fill: 1.000% (retained sits inside top 1% tier, not top 0.1%).
- tier target: unmet (required_unique=500000; actual 87005).
- MMR selection: shortlist_k=2500 (target 2500) from tier fraction 0.09 (tier_limit=7830).
- scores (min/med/avg/max): 17.21 / 18.02 / 18.28 / 21.65.
- lengths (min/med/avg/max): 17 / 19.0 / 19.1 / 20.
- diversity: pairwise median 2.77 (top 2.80, delta -0.02); overlap 99.6% (2 swaps).
- score deltas (p10/median): +0.00 / +0.00 (MMR did not trade away score).
- mining tail slope: about 0.108 delta unique / delta generated (yield flattening).
- padding audit: 100% best-hit overlap with intended core; core offsets concentrated at 0-5.

### cpxR_MANWWHTTTAM
- generated: 500000; candidates_with_hit: 500000; eligible_raw: 483597.
- eligible_unique: 120124 (about 24% of raw; duplication factor about 4.0x).
- retained: 500; tier fill: 1.000% (retained inside top 1% tier).
- tier target: unmet (required_unique=500000; actual 120124).
- MMR selection: shortlist_k=2500 (target 2500) from tier fraction 0.09 (tier_limit=10811).
- scores (min/med/avg/max): 8.51 / 9.00 / 9.19 / 11.56.
- lengths (min/med/avg/max): 15 / 18.0 / 17.6 / 20.
- diversity: pairwise median 3.39 (top 3.41, delta -0.01); overlap 99.0% (5 swaps).
- score deltas (p10/median): +0.00 / +0.00.
- mining tail slope: about 0.121 delta unique / delta generated.
- padding audit: 100% best-hit overlap with intended core; core offsets spread across 0-9.

## Behavioral interpretation (Stage-A evaluation)
- Tier-target math vs budget: target_tier_fraction=0.001 implies 500000 eligible uniques are needed; both motifs fall short, so retention spills beyond the 0.1% tier (warnings are expected). Tier fill shows retention still stays within the top 1% of ranked uniques.
- MMR effect: overlap above 99% and near-zero pairwise deltas indicate MMR makes only minimal swaps at alpha=0.9 with a 2500 shortlist. This preserves score distribution (no p10/median loss) but yields limited diversity gains. If diversity is a priority, reduce alpha, widen the shortlist, or relax uniqueness (sequence instead of core) and/or increase mining budgets.
- Mining yield: tail slope about 0.11-0.12 suggests diminishing returns at 500k candidates; more mining might still help but at decreasing efficiency.
- Length and GC: retained TFBS lengths align with the requested range and show mild score-length relationships; GC is tracked in plots to spot bias.
- Core alignment: 100% overlap of best hit with intended core indicates FIMO scoring aligns with the sampled PWM core placement (no systematic shifts from flanks).

## Stage-A plots (interpretation)
Plot interpretation lives in the sampling guide:
`../guide/sampling.md#how-to-read-stage_a_summary-three-figures`.

## Audit takeaways
- Stage-A behavior is consistent with the documented semantics and current code paths.
- The demo run is score-dominated (alpha=0.9, shortlists from top 9%); diversity gains are minimal, but score integrity is preserved.
- The tier-target warning is expected given max_candidates=500000 and core-level uniqueness; the retained set still sits in the top 1% of the ranked pool.
