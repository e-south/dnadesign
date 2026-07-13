---
doc_id: study-eco1-rt-repack-fixed-backbone-method
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-12
---

## Fixed-Backbone Method

This study adapts an AI-guided RT redesign pattern to Eco1 RT for downstream
sponging workflows. The method is computational and contract-first:

1. Choose one structure authority and chain policy.
2. Map every designable residue into a canonical Eco1 RT numbering system.
3. Define complete generation policies from protected positions, open
   positions, residue-specific alphabets, and explicit missing-backbone handling.
4. Generate fixed-backbone sequence samples with a declared MPNN backend.
5. Normalize and deduplicate complete candidate sequences.
6. Review fold and local backbone geometry with declared metrics.
7. Select a mutation-set-diverse panel from rows that pass the declared checks.
8. Emit a candidate handoff only when every upstream artifact is present and
   hash-linked.

The motivating source method is Tao et al., Nature Biotechnology 2026,
DOI `10.1038/s41587-026-03149-6`. This study uses the computational pattern,
not the prime-editing objective.

### Literature Roles

The method intentionally separates source roles instead of treating any paper as
a monolithic recipe:

- Tao et al. provides the fixed-backbone RT redesign pattern: preserve
  functional and conserved residues, sample fold-compatible RT sequences, and
  filter candidates structurally. It does not make the entire Mestre roster the
  MSA denominator.
- ProteinMPNN provides the backend request format for this first sampling path:
  helper-compatible parsed PDB JSONL, assigned-chain JSONL, fixed-position
  JSONL, explicit seed/temperature fields, and omitted-amino-acid policy. It
  does not define Eco1 mask policy, decide which residues are protected, or
  evaluate function.
- Mestre et al. provides the retron RT source ontology: use Ec86 RT clade 9 as
  the broad homolog panel and II-A3/`42_1` as the Eco1-family panel.
- Simon et al. provides RT-region and motif annotation grammar for figures and
  audits. Annotation tracks are not automatically mask-authoritative.
- Wang et al. provides the Eco1/Ec86 structural context and direct-contact
  priors. The current mask consumes the audited direct-contact rows as active
  protection evidence; RT1-RT7 intervals remain review labels, not blanket mask
  authority.
- Inouye et al. 1999 and Inouye et al. 2004 provide Ec86-specific evidence
  that the C-terminal/thumb region participates in cognate primer-RNA
  recognition. This supports a visible C-terminal/thumb review axis. It does
  not make thumb mutation a conservative default and does not provide a
  strand-displacement score.

### Method Posture

Treat ProteinMPNN/LigandMPNN output as fold-compatible sequence proposals, not
as proof of improved stability or function. A candidate becomes useful only
after it passes mask audit, deduplication, structural QA, and downstream
promotion checks.

### ProteinMPNN Usage

This study uses ProteinMPNN as a fixed-backbone inverse-folding sampler. The
input backbone is the selected protein-only Ec86 RT chain from 7V9U. Eco1
policy determines which canonical residues are protected before ProteinMPNN is
called; ProteinMPNN receives only the chain-local fixed-position list needed to
sample the remaining mapped residues.

The executable path follows the public ProteinMPNN command-line workflow:

- parse the protein-only PDB into ProteinMPNN JSONL input;
- declare the designed chain;
- provide fixed positions through the ProteinMPNN fixed-position JSONL;
- run `protein_mpnn_run.py` with explicit seed, sampling temperature,
  `num_seq_per_target`, and omitted amino acids.

ProteinMPNN positions are chain-local and 1-indexed by sequence order, not raw
PDB residue ids. `dnadesign.thread.adapters.proteinmpnn` therefore maps
canonical Eco1 positions to ProteinMPNN chain positions before writing fixed
positions. Terminal Eco1 positions without 7V9U backbone coordinates are not
sent as mutable fixed-backbone positions.

The active v3 request uses official ProteinMPNN commit
`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, an explicit tool root, seeds
`101`, `202`, and `303`, sampling temperatures `0.1` and `0.3`, and
`num_seq_per_target: 56` for 336 requested sequences per policy. Peripheral
residues use the public `--omit_AA_jsonl` input, and v3 uses global
`--omit_AAs C`. An open WT cysteine can therefore be forced to change. The
output is a sequence-proposal table, not a stability measurement, fold check,
or functional assay.

Methods-ready wording:

> ProteinMPNN was run as a fixed-backbone inverse-folding sampler on the
> selected Ec86 RT backbone. NAxxH, YADD, and VTG contexts, direct
> retained DNA/RNA contacts, Wang thumb-track positions, mapped residues
> 255-311, and declared conserved/core positions were fixed before sampling.
> Open positions and residue-specific allowed alphabets were supplied in
> chain-local 1-indexed ProteinMPNN coordinates. New cysteine was omitted; an
> open WT cysteine could therefore be forced to change; and peripheral
> alternatives were limited to MSA-observed residues without new D/E, P, or G.
> Complete sequences were
> generated with declared seeds, sampling temperatures, and sample counts, then
> deduplicated and passed to fold checking.

The study premise is:

```text
This study compares WT Eco1 RT with complete ProteinMPNN-designed sequences
that either repack distal scaffold positions, redesign a non-acidifying,
MSA-supported peripheral nucleic-acid-facing shell, or do both, while keeping
declared catalytic, direct-contact, Wang thumb-track, and mapped residues
255-311 fixed and requiring preserved predicted local backbone geometry.
```

The first pass should not jointly redesign the RT and lnRNA/pretroDNA substrate.
Use a constant downstream substrate context until RT-only candidate behavior is
understood.

### Stage Contracts

| Stage | Input contract | Output contract | Owner |
| --- | --- | --- | --- |
| Structure authority | Selected PDB/mmCIF, chain policy, reference sequence hash. | `BackboneBundle`, `ResidueMap`. | study then `thread` |
| Evidence profiles | Residue map plus MSA/contact source declarations. | `ConservationProfile`, `ContactProfile`. | study policy, `thread` normalization |
| Generation policy | Evidence profiles plus study protection and alphabet rules. | Fixed positions, open positions, per-residue alphabets, and policy hash. | study policy with generic ProteinMPNN request mechanics |
| Sampling | One complete generation policy plus backend request. | Complete ProteinMPNN sequence rows with one policy id. | `thread` contracts; `infer` optional execution provider |
| Candidate pool | Sample rows plus policy provenance. | Deduplicated complete-sequence table. | `thread` mechanics, study provenance checks |
| Fold QA | Candidate table plus fold runtime declaration. | `FoldCheckReport` and normalized model set. | `thread` normalization; SCC runtime |
| Local geometry and selection | Candidate and fold rows plus named regions and selected-panel policy. | Local-RMSD table and one eight-row selected panel. | study selection policy |
| Downstream handoff | Selected protein rows and hashes. | `CandidateHandoff`. | `thread` bundle, then RT-lnRNA acceptance |

Every stage accepts only the previous stage's declared artifact, never an
ad-hoc reconstruction from filenames or transient notebook state.

### Study Boundary

Eco1 profile choices, catalytic protection, structural-source selection, and
candidate-batch policy stay study-owned. Generic fixed-backbone request,
sample-ingest, and candidate-table mechanics may live in `thread` only when
they are free of Eco1 biology and have their own contract tests.

### Execution Boundary

Do not hide model execution behind an implicit run-all framework. Eco1 can call
the generic ProteinMPNN adapter with an explicit tool root and request manifest,
then call generic candidate-table construction after sample ingest. The v3
policy, ProteinMPNN, candidate-pool, ColabFold, fold-review, and selection
artifacts are materialized and keyed to the v3 policy hash.

Use `implementation-roadmap.md` for the active runtime order. That
page owns stage order, owner boundaries, canonical commands, and fail-fast
conditions.
