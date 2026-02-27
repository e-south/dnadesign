## Stage-A Summary Plot Doc Alignment Design


### Contents
- [Context](#context)
- [Goal](#goal)
- [Scope](#scope)
- [Non-Goals](#non-goals)
- [Recommended Approach (A)](#recommended-approach-a)
- [Proposed Changes](#proposed-changes)
- [1) Canonical narrative in sampling.md](#1-canonical-narrative-in-samplingmd)
- [2) Remove interpretive duplication elsewhere](#2-remove-interpretive-duplication-elsewhere)
- [3) Drift prevention note](#3-drift-prevention-note)
- [Validation](#validation)
- [Risks](#risks)
- [Rollout](#rollout)

### Context
Stage-A produces a three-figure plot bundle (`stage_a_summary`) that is used to explain sampling quality, yield, and diversity behavior. The current narrative is scattered across multiple docs, and the didactic takeaways can drift from what the plots actually show. The goal is to make the documentation semantically aligned with the visuals and keep it aligned over time.

### Goal
Create a single authoritative "How to read `stage_a_summary`" section in `sampling.md` that matches the current plot semantics and labels, and make other docs link to it rather than duplicating interpretation.

### Scope
- Documentation updates only.
- No code changes.
- No new tooling or automated tests.

### Non-Goals
- Changing plot behavior or metrics.
- Adding new plots or annotations.

### Recommended Approach (A)
Move the full interpretive narrative into `src/dnadesign/densegen/docs/concepts/sampling.md` as a dedicated section and replace interpretive prose elsewhere with links to that section.

### Proposed Changes
#### 1) Canonical narrative in sampling.md
Add a section titled "How to read `stage_a_summary` (three figures)" with three subsections:

1. **`stage_a_summary__<input>.png` (tiers)**
   - What it shows: score distribution of eligible unique cores, tier cutoffs labeled by configured percentages, retained cutoff as the minimum retained score, and retained TFBS length histogram.
   - Question it answers: how close the retained pool is to the high-score frontier and how deep selection went to fill `n_sites`.

2. **`stage_a_summary__<input>__yield_bias.png` (yield + entropy)**
   - What it shows: stepwise yield across Generated → Eligible → Unique core → MMR pool → Retained, and core positional entropy for diversified sequences only.
   - Clarify x-axis labels as IUPAC consensus per PWM position.
   - Question it answers: where filtering happened and which motif positions remain variable in the final pool.

3. **`stage_a_summary__<input>__diversity.png` (outcome + mechanism)**
   - What it shows: pairwise distance ECDF for Top Sequences vs Diversified Sequences (outcome) and selection-time nearest distance vs Score / PWM consensus score (mechanism).
   - Clarify Δdiv (median pairwise distance gain) and ΔJ (objective gain).

Add a short glossary for the inline metrics that appear in the plots:
- **Δdiv (median)**: median pairwise distance change (diversified − top)
- **ΔJ**: MMR objective gain (diversified − top)
- **overlap**: fraction of shared sequences between Top and Diversified sets

#### 2) Remove interpretive duplication elsewhere
- `src/dnadesign/densegen/docs/reference/outputs.md`: keep plot inventory only; add a link to the sampling guide for interpretation.
- `src/dnadesign/densegen/docs/tutorials/demo_pwm_artifacts.md`: replace interpretive text with a short link to the sampling guide section.
- `src/dnadesign/densegen/docs/dev/audit_2026-01-30_stage_a.md`: keep run notes but remove plot interpretation; link to the sampling guide section.

#### 3) Drift prevention note
Add a short "Doc-plot sync checklist" at the end of the new sampling.md section listing the exact plot labels that must match the docs:
- Top Sequences
- Diversified Sequences
- Score / PWM consensus score
- IUPAC consensus labels on entropy x-axis
- Yield stage labels: Generated → Eligible → Unique core → MMR pool → Retained

### Validation
- Run `uv run dense plot --only stage_a_summary` and visually confirm labels and semantics match the sampling guide section.
- Check all updated docs for consistent terminology and links.

### Risks
- Future plot label changes can reintroduce drift unless the checklist is followed.

### Rollout
- Apply doc updates in a single commit.
- Announce the new canonical location for plot interpretation to avoid future duplication.
