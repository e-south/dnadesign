---
doc_id: study-eco1-rt-repack-status
surface: study-record
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-13
status_surface: record-only
---

## Eco1 RT Repack Status

### Study Premise

This study compares WT Eco1 RT with complete ProteinMPNN-designed sequences
that either repack distal scaffold positions, redesign a non-acidifying,
MSA-supported peripheral nucleic-acid-facing shell, or do both, while keeping
declared catalytic, direct-contact, Wang thumb-track, and mapped residues
255-311 fixed, requiring preserved predicted local backbone geometry, and
reporting substitutions at the Wang-described alpha-1 protomer interface for
review.

The study does not claim improved activity, affinity, processivity, strand
displacement, or safety.

### Subdeliverable Premises

| Subdeliverable | Purpose |
| --- | --- |
| Structure and residue map | Define Eco1 numbering and retained DNA/RNA geometry. |
| Conservation and generation-policy records | Define fixed positions, open positions, and allowed amino acids. |
| ProteinMPNN pool | Supply complete sequence proposals, not mutation parts or functional predictions. |
| ColabFold and local RMSD | Identify predicted global or regional structural disruption. |
| Chemistry and MSA audit | Describe mutation geography, charge events, and homolog support without scoring activity. |
| Panel selection | Reduce mutation-set overlap within the distal, peripheral, and combined groups. |
| Notebook and sequence export | Expose the trace, structures, and complete protein hypotheses without adding criteria. |
| ESMC and SAE | Provide optional model context; neither is essential selection evidence. |

The minimal evidence is the residue authority, protected/open policy manifest,
complete candidate sequences, fold and local-geometry measurements, mutation and
charge annotations, regional MSA support, the selection trace, and the complete
selected sequences. Other outputs are supporting audit context.

### Current State

The active generation policy, ProteinMPNN sampling, ColabFold folding,
local-structure review, and eight-sequence selection are materialized under
`outputs/thread/generation_policies_v3/`.

The selection flow is:

1. `1007` accepted complete ProteinMPNN sequences;
2. `738` sequences retaining the generation constraints and local C-alpha RMSD
   at or below the declared `2.5 A` cutoff in every non-distal review region
   after one global mapped fit;
3. three design groups containing `335` distal, `226` peripheral, and `177`
   combined rows; group assignment does not remove rows;
4. eight selected sequences: two distal, three peripheral, and three combined.

The two distal rows contain `16-17` substitutions and no peripheral-shell
changes. The three peripheral rows contain `32-35` substitutions, all in the
peripheral shell, with shell charge changes from `+7` to `+11`. The three
combined rows contain `47-57` substitutions, including `29-38` peripheral
changes, with shell charge changes from `+6` to `+10`. These are planned
experimental comparisons, not quality tiers or biological replicates.

The distal policy opens 25 positions more than `10 A` from retained DNA/RNA.
They are concentrated at residues `3-32`, with three additional positions at
`61`, `75`, and `79`. This is an N-terminal-enriched distant-repacking control
for general repacking and fold preservation, not a direct strand-displacement
hypothesis. Its value is comparative: it separates broad repacking effects from
the peripheral nucleic-acid-facing intervention.

Wang places alpha-1 at the interface between two RT-msDNA protomers and reports
that R13A disrupts that contact. The same experiment retained msDNA production
and antiphage defence, and the paper reports that alpha-1 is not conserved in
the related Sen2 and Eco9 RT complexes. R13 is therefore an interface
annotation, not a functional hard gate. It was open in the distal and combined
policies because it is `17.993 A` from retained DNA/RNA, outside the thumb track
and mapped `255-311` context, and below the declared clade-9 conservation
threshold. Four selected rows retain WT R13 and four substitute it;
the exact residue state is reported without changing eligibility or rank.

The review notebook exposes core evidence, communication visuals, and optional
model checks as separate evidence sets. One retained-complex browser contains
the active mask evidence, design spaces, and RT annotation spans. The visible
communication set contains the residue-position map, structural screen, and
selected-mutation map. ChimeraX movies enter notebook
navigation only after they are rendered.
Protein-DNA-RNA views use gold DNA and salmon RNA across each chain's backbone
and nucleotide representation. ChimeraX uses ladder nucleotides; py3Dmol uses a
flat coordinate-derived backbone ribbon with attached base spokes. Protein
surfaces are protein-only and are off by default in the interactive notebook.
The protected-category movie uses `55%` ChimeraX surface transparency so the
gray protein cartoon remains visible and shows atoms only for the active
category's side chains. The qualitative Coulombic movie uses an opaque protein
surface and one fixed `-10` to `+10 kcal/(mol e)` scale at `298 K`.

### Structure And Sequence Authority

- Structure authority is `ec86kit_7v9u_protomer1`: PDB `7V9U`, RT chain `A`,
  retained DNA chain `D`, and retained RNA chains `E/F`.
- The residue map contains `309` mapped positions. Residues `1`, `2`, and
  `312-320` lack mapped backbone coordinates.
- The selected conservation source is `ec86_clade9_conservation_v1`. The
  II-A3/`42_1` profile is comparison context.
- Protected motif-context windows are `99-115`, `189-204`, and `237-251`; the
  exact NAxxH, YADD, and VTG anchors are `105-109`, `195-198`, and `243-245`.
  ProteinMPNN fixed every residue in each broader window. The exact anchors are
  labeled inside those windows so motif extent is not confused with the
  precautionary flanks. The flank widths are declared study choices, not known
  functional boundaries.
- Direct retained DNA/RNA contacts at or below `5 A` and Wang thumb-track
  positions `238`, `239`, `240`, `249`, `257`, `261`, `264`, and `298` are
  fixed.
- Mapped residues `255-311` are fixed. Residues `230-254` are not part of this
  fixed set; their substitutions remain visible in mutation and local-structure
  review.

### Generation Policies

| Policy | Open positions | Accepted sequences |
| --- | ---: | ---: |
| `distal_scaffold_repack_v1` | 25 distal | 335 |
| `near_dna_rna_acid_free_v1` | 59 peripheral | 336 |
| `combined_near_acid_free_plus_distal_v1` | 84 peripheral and distal | 336 |

Peripheral positions are more than `5 A` and at or below `10 A` from retained
DNA/RNA, outside the shared fixed set. The combined policy opens peripheral and
distal positions in one ProteinMPNN request. Mutations are not combined across
policy outputs.

ProteinMPNN requests use the public `--omit_AA_jsonl` sidecar for
residue-specific peripheral alphabets and `--omit_AAs C` for the global
no-cysteine rule. Peripheral alternatives are MSA-observed; new D/E and new
P/G substitutions are excluded. WT acidic, proline, or glycine residues may be
retained.

Open-site WT retention is calculated over the `25`, `59`, or `84` designable
positions in each policy. Its vertical bands are the expected one-residue
increments for those finite open sets, not a scoring or plotting defect.

C233 was open in the proximal policies, and the global no-cysteine rule omitted
its WT residue. Proximal-policy sequences therefore share a C233 substitution.
This is a declared generation constraint and a source of panel overlap, not a
protected-position violation or functional signal.

All selected peripheral or combined rows change C233 and G254; five change
K230. The two distal rows do not change these positions. Positions `230-254`
form a designable boundary outside the fixed `255-311` set, and the regional
mutation plot reports that boundary separately.

### Selection Method

The visible flow separates eligibility from experimental design:

1. accepted complete sequences;
2. one constraint and local-geometry rule across all non-distal review regions;
3. assignment of passing rows to the three design groups;
4. mutation-set selection of two distal, three peripheral, and three combined
   sequences.

Protected-position, direct-contact, Wang-track, acidic-gain, and proximal MSA
checks remove no active rows because those constraints were enforced during
generation. They remain audited invariants rather than decorative funnel
stages.

Within each group, the first pair has the greatest mutated-position Jaccard
distance, with exact-substitution distance second. The third peripheral or
combined row maximizes its minimum distance from that pair. Charge-event
counts, MSA support, local RMSD, fold metrics, and sequence hash are later
tie-breakers. Alpha-1 and R13 substitutions are reported annotations. Net
charge and total mutation count do not rank rows.

Across all eight rows, the panel covers `72` mutated positions and `144` exact
substitutions. No position or exact substitution is shared by all eight because
the distal and peripheral policies open different residue sets. That global
result overstates within-policy diversity. Within the distal pair, `11`
positions are shared and position-set distance is `0.500`. The peripheral trio
shares `22` positions and has a minimum position-set distance of `0.366`. The
combined trio shares `35` positions and has a minimum position-set distance of
`0.305`. The panel is mutation-set-diverse, not orthogonal.

### Evidence Limits

- Wang et al. 2022 and `7V9U` support the Ec86 RT-msDNA/RNA geometry, direct
  contacts, and electropositive-surface prior. They do not validate a charge
  optimum or activity score. Wang also places the N-terminal alpha-1 region at
  an RT-msDNA protomer interface. Its R13A experiment disrupts the protomer
  contact while retaining msDNA production and antiphage defence, and alpha-1
  is not conserved in the related Sen2 and Eco9 complexes. This supports
  reporting alpha-1 substitutions, not excluding every R13 substitution. It
  also does not establish that every alpha-1 substitution is harmless, so
  distal does not mean function-free.
- Inouye et al. 1999 supports caution around the C-terminal 91-residue
  primer-template recognition context. Inouye et al. 2004 directly supports the
  `255-320` primer-recognition RNA-binding fragment.
- Tao et al. 2026 supports constraint-first RT generation and structural
  review. It does not validate the Eco1 `5-10 A` shell or the `2.5 A` cutoff.
- Kabsch 1976/1978 supports the global rigid-body fit, not the local cutoff.

Primary citations and their narrow study roles are listed in
`contexts/selection-hardening-dev-spec.md` and
`.agents/skills/eco1-rt-repack-status/references/external-sources.md`.

### Selected Panel

The selected panel contains eight sequences: two distal, three peripheral, and
three combined. Exact candidate ids, ranks, mutation counts, and sequence hashes
are read from `selection/candidate_selection_panel.parquet`; they are not
duplicated here.

### Twist Handoff

The synthesis handoff contains all eight selected full-length CDS designs. Each
record is `963 bp`, including the WT stop
codon, and translates to the corresponding canonical `320 aa` fold input.
Unchanged amino acids retain the authoritative WT codon; changed amino acids use
one recorded E. coli codon policy. Independent vendor codon optimization is
disabled.

The bundle contains one upload CSV, one FASTA, and one annotated GenBank file
per candidate. GenBank variation features use compact amino-acid labels such as
`A47K`. The manifest reports GC fraction, maximum homopolymer length, repeated
20-mer count, and internal BsaI/BsmBI checks for every ordered sequence.

The exact sequences are ready for Twist upload, complexity review, and a live
quote. At Twist's advertised `$0.07/bp` floor, eight `963-bp` fragments total
about `$539` before shipping or complexity charges. They are not cloning-ready
because assembly flanks and junctions have not been declared or screened.

Handoff manifest:
`outputs/thread/generation_policies_v3/twist_handoff/twist_handoff_manifest.yaml`.

### Validation Commands

```bash
uv run pytest -q src/dnadesign/thread/tests/adapters/proteinmpnn
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/twist_handoff
uv run marimo check src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/notebooks/eco1_review_deliverables.py
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run python -m dnadesign.devtools.docs.checks --repo-root .
```
