# 2026-04-19 retron P4 hairpin and YIU executive summary

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-19

## Scope

This note consolidates the findings that were previously split across:

- `src/dnadesign/cruncher/docs/dev/audits/2026-04-19-retron-p4-hairpin-variant.md`
- `2026-04-19-yiu-retron-mismatch-bulge-audit.md`

The goal is a single high-signal summary of:

- what the local retron `26 / 43 / 45 / 46` data actually support
- what those variants imply about stem length, WT-stem preservation, bulges, and rescue logic
- how that biological design logic does and does not map onto current `cruncher yiu`

## Executive summary

The strongest local conclusion is not that every variant after `retron-26` failed equally. The stronger conclusion is:

- `retron-26` is the only strong working TetR-sponge baseline
- `retron-43` is the clear stem-extension failure case
- `retron-45` is a weak partial rescue
- `retron-46` is weaker than `45`
- neither `45` nor `46` comes close to restoring `26`-level function

The most coherent mechanistic read is:

- extending or over-stabilizing the P4-region hairpin impairs productive reverse transcription
- preserving more WT-like scaffold logic helps somewhat
- adding a local irregularity or bulge is a plausible rescue move, but in this local dataset it is not sufficient
- the design bottleneck is not just decoy sequence identity; it is compatibility with RT processivity, strand displacement, and mature RT-DNA accumulation

For YIU, the main conclusion is boundary clarity:

- current YIU is mismatch-centric
- it models mismatch position within a fixed `4 nt` junction window
- it does not model true bulged or scaffold-altered payload topology
- "middle-only mismatch pool" is not the same thing as a "mid-payload bulge"

## Highest-signal findings

### 1. Local assay ranking is stronger than the slide-only story

The July 7 four-way reader benchmark is the clearest local result set.

At `10 h`, design-normalized fold change under IPTG is:

| Variant | `5 uM IPTG` | `50 uM IPTG` | `500 uM IPTG` | `200 nM aTc` |
| --- | ---: | ---: | ---: | ---: |
| `retron-26` | `1.038` | `3.666` | `8.089` | `5.783` |
| `retron-43` | `0.999` | `1.139` | `1.038` | `6.762` |
| `retron-45` | `1.076` | `1.331` | `1.466` | `6.112` |
| `retron-46` | `1.014` | `1.060` | `1.211` | `5.488` |

This means:

- all constructs still respond to direct `aTc`
- only `26` yields strong IPTG-responsive sponge behavior
- `45` is directionally better than `43`
- `46` is also weak and trails `45`

The June 22 two-way benchmark independently validates the same `26 >> 43` split:

- `26` at `12 h`: `4.960x` at `50 uM IPTG`, `4.456x` at `500 uM IPTG`
- `43` at `12 h`: `1.041x` at `50 uM IPTG`, `1.010x` at `500 uM IPTG`

### 2. The local design lesson is about topology and processivity, not only motif sequence

Across the decks and local figures:

- `retron-26` is the proven compact baseline
- `retron-43` extends the stem and loses strong sponge behavior
- `retron-45` and `retron-46` are rescue attempts that preserve more WT-like scaffold logic and/or introduce a deliberate irregularity

The best working interpretation is:

- simple stem extension is harmful
- WT-like scaffold preservation helps a little
- controlled irregularity can help slightly
- rescue remains weak if the design still pushes the ncRNA into a poor RT-processing regime

### 3. Slide labels and figure-source labels are not perfectly stable

The later progress-report slide lineage labels `46` as the explicitly bulged variant.

The archived SVG figure lineage instead labels `45` as:

- `Modified lnRNA with bulge at base of stem`

This is best treated as annotation drift across figure versions. It is a documentation risk, not a reason to discard the broader rescue interpretation.

### 4. YIU and retron bulge logic should not be conflated

Current YIU docs, schema models, and tests are explicit:

- the junction is always an internal `4 nt` window
- `candidate_positions` must be a subset of `0..3`
- `candidate_positions=[1,2]` is a middle-only mismatch pool
- legacy `bulge_mask`, `split`, and related topology keys are rejected

So YIU can represent:

- edge versus middle mismatch policies
- one- versus two-mismatch plans
- ligation-aware ranking over those plans

YIU cannot represent:

- a real bulged payload topology
- a scaffold-preserving ncRNA redesign
- a retron-style WT-stem-versus-extended-stem rescue problem

## Variant readout

| Variant | Best local role | Functional read |
| --- | --- | --- |
| `retron-26` | working compact engineered baseline | strong TetR sponge |
| `retron-43` | stem-extension stress test | clear failure |
| `retron-45` | WT-like or bulge-adjacent rescue attempt | weak partial rescue |
| `retron-46` | later irregularity or bulge rescue attempt | weak, below `45` |

The practical ranking is:

- `26` strong
- `45` weak
- `46` weak
- `43` weakest

That is enough to reject the stronger rescue claim while still preserving the weaker directional one.

## Conclusions

### Biological design conclusion

The data support a compact rule:

- avoid simple P4 stem extension
- preserve WT-like architecture where possible
- use bulges or mismatches only as targeted processivity interventions
- judge success by RT-DNA or reporter output, not by ViennaRNA-predicted fold
  appearance alone

The most defensible route to success is to stay close to a `26`-like scaffold and introduce only minimal, hypothesis-driven irregularity.

### Software and product conclusion

The right YIU stance is to stay explicit about abstraction boundaries.

Short term:

- keep YIU mismatch-only
- keep "middle-only" terminology tied to offsets inside the ligation window
- do not imply YIU already covers bulged-topology reasoning

If topology-aware design becomes necessary later, the safer path is a separate contract or extension with first-class topology fields rather than overloading mismatch coordinates.

## Source hierarchy

### Highest-confidence local evidence

- `reader/experiments/2025/20250622_retron_Eco1_26_43_benchmark/outputs/artifacts/fold_change__single_reporter.transform_fold_change/table.parquet`
- `reader/experiments/2025/20250707_retron_Eco1_26_43_45_46_benchmark/outputs/artifacts/fold_change__single_reporter.transform_fold_change/table.parquet`
- `reader/experiments/2025/20250707_retron_Eco1_26_43_45_46_benchmark/outputs/artifacts/ratio_reporter_normalizer.transform_ratio/df.parquet`

### Supporting local context

- `progress_reports/250626_ejs.pptx`
- `progress_reports/250708_ejs.pptx`
- `progress_reports/250808_ejs.pptx`
- `progress_reports/250925_ejs.pptx`
- `figures/archived/year_six/retrons_2025b.svg`
- `figures/archived/year_six/retrons_2025c.svg`

### YIU contract sources

- `dnadesign/src/dnadesign/cruncher/docs/reference/yiu_spec.md`
- `dnadesign/src/dnadesign/cruncher/docs/guides/yiu_workflow.md`
- `dnadesign/src/dnadesign/cruncher/src/yiu/spec_rendering_models.py`
- `dnadesign/src/dnadesign/cruncher/tests/cli/test_yiu_cli.py`
- `dnadesign/src/dnadesign/cruncher/tests/yiu/test_payload_rendering.py`

### Literature anchors

- Palka et al. 2022: `10.1093/nar/gkac177`
- Crawford et al. 2025: `10.1093/nar/gkae1199`
- Wang et al. 2022: `10.1038/s41564-022-01197-7`
- Lanciault and Champoux 2006: `10.1128/JVI.80.5.2483-2494.2006`
- Ohshima and Wells 1997: `10.1074/jbc.272.27.16798`
- Canceill, Viguera, and Ehrlich 1999: `10.1074/jbc.274.39.27481`

## Open questions

- The exact meaning of the `l=` labels remains implicit in the slide corpus.
- The figure lineage still shows some label drift around which rescue variant carried the explicit bulge annotation.
- The local data support only weak rescue for `45/46`, not a restored functional class.
- Current YIU test naming can suggest "bulged" behavior where the underlying model is still only middle-position mismatch selection.
