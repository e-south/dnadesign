# DenseGen Stage-A audit (2026-01-30)

- Scope: Stage-A PWM sampling behavior and diagnostics for demo_meme_two_tf.
- Branch: densegen/cruncher-refine.
- Workspace: src/dnadesign/densegen/workspaces/demo_meme_two_tf.
- Command: uv run dense stage-a build-pool --fresh (config auto-detected).
- Inputs: pwm_artifact_set (LexA + CpxR).

## Stage-A intent and rationale (code-aligned)
- Stage-A builds TF-specific TFBS pools from PWM artifacts using FIMO log-odds scoring (forward strand only, thresh=1.0).
- Eligibility requires at least one hit and best_hit_score > 0; sequences are deduplicated by core (matched_sequence) before ranking.
- Ranking is score-first with deterministic tie-breaks (lexicographic sequence).
- Diagnostic tiers (0.1%, 1%, 9%, rest) and tier-targeted mining provide budget control and early warnings.
- MMR selection trades score vs diversity on tfbs_core with PWM-tolerant weighted Hamming distance; shortlist pool built from a tier ladder for headroom.

## Stage-A settings used in demo_meme_two_tf
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
- diversity: pairwise median 2.77 (baseline 2.80, delta -0.02); overlap 99.6% (2 swaps).
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
- diversity: pairwise median 3.39 (baseline 3.41, delta -0.01); overlap 99.0% (5 swaps).
- score deltas (p10/median): +0.00 / +0.00.
- mining tail slope: about 0.121 delta unique / delta generated.
- padding audit: 100% best-hit overlap with intended core; core offsets spread across 0-9.

## Behavioral interpretation (Stage-A evaluation)
- Tier-target math vs budget: target_tier_fraction=0.001 implies 500000 eligible uniques are needed; both motifs fall short, so retention spills beyond the 0.1% tier (warnings are expected). Tier fill shows retention still stays within the top 1% of ranked uniques.
- MMR effect: overlap above 99% and near-zero pairwise deltas indicate MMR makes only minimal swaps at alpha=0.9 with a 2500 shortlist. This preserves score distribution (no p10/median loss) but yields limited diversity gains. If diversity is a priority, reduce alpha, widen the shortlist, or relax uniqueness (sequence instead of core) and/or increase mining budgets.
- Mining yield: tail slope about 0.11-0.12 suggests diminishing returns at 500k candidates; more mining might still help but at decreasing efficiency.
- Length and GC: retained TFBS lengths align with the requested range and show mild score-length relationships; GC is tracked in plots to spot bias.
- Core alignment: 100% overlap of best hit with intended core indicates FIMO scoring aligns with the sampled PWM core placement (no systematic shifts from flanks).

## Demo narrative and plots (how they support the story)
The demo explicitly calls Stage-A before Stage-B; the key diagnostic plot is stage_a_summary:

1) stage_a_summary__lexA_cpxR_artifacts.png (tiers and length)
- Left panels: per-TF score distributions for eligible uniques with the retained subset shaded.
- Tier cutoff markers (0.1%, 1%, 9%) and the retained cutoff show where the final pool sits.
- Right panel: retained TFBS length counts confirm the requested 15-20 nt range.

2) stage_a_summary__lexA_cpxR_artifacts__yield_bias.png
- Left: yield funnel (Generated -> Has hit -> Eligible -> Unique core -> MMR pool -> Retained) with duplication pressure, MMR headroom, and mining tail slope annotations.
- Right: scatter of retained score vs length, colored by GC fraction, plus rho(score,length) and hit overlap.
- Narrative use: validates that Stage-A is converting candidates into a high-quality pool without obvious length/GC bias or core misalignment.

3) stage_a_summary__lexA_cpxR_artifacts__diversity.png
- Left: pairwise core distance ECDF (Top-score vs MMR), with Δdiv (median gain), ΔJ (objective gain), and overlap annotations.
- Right: score vs selection-time nearest distance (MMR contribution), with scores normalized by pwm_max_score
  in FIMO score scale.
- Narrative use: separates the final diversity outcome (left) from the selection-time tradeoffs that produced it (right).

The demo suggests running dense plot --only stage_a_summary right after stage-a build-pool so you can link the warnings (tier target unmet) to visual evidence of tier placement and diversity tradeoffs.

## Audit takeaways
- Stage-A behavior is consistent with the documented semantics and current code paths.
- The demo run is score-dominated (alpha=0.9, shortlists from top 9%); diversity gains are minimal, but score integrity is preserved.
- The tier-target warning is expected given max_candidates=500000 and core-level uniqueness; the retained set still sits in the top 1% of the ranked pool.
