---
doc_id: study-eco1-rt-repack-status
surface: study-record
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-15
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
displacement, safety, or a monomeric RT-msDNA assembly state.

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

Wang describes a homodimer of two RT-msDNA protomers. F10 and R13 in alpha-1 of
each RT contact msDNA in the other protomer. R13A disrupted this interaction
while retaining msDNA and the tested antiphage phenotype. Related Sen2 and Eco9
RT-msDNA complexes were monomeric. These results identify R13A as a tested
interface-disrupting substitution; they do not establish the assembly state or
function of other F10 or R13 substitutions.

The structure and fold workflow uses one RT chain with its retained DNA/RNA as
a coordinate reference. It does not predict oligomeric state. None of the eight
selected sequences contains R13A, and no R13A sequence occurs anywhere in the
`1007`-sequence v3 pool. Three selected rows retain WT F10/R13, one retains WT
R13 with F10E, three contain R13K, and one contains R13E. Exact F10/R13 states
are reported without changing eligibility or rank. The selected sequences must
not be described as designed monomers.

The review notebook exposes core evidence, communication visuals, and optional
model checks as separate evidence sets. One retained-complex browser contains
the fixed positions, open design spaces, and RT annotation spans. The
communication set contains the residue-position map, structural screen,
selected-mutation map, and any rendered movies. Absent movies do not enter the
notebook choices.
Protein-DNA-RNA views use gold DNA and salmon RNA across each chain's backbone
and nucleotide representation. ChimeraX uses ladder nucleotides; py3Dmol uses a
flat coordinate-derived backbone ribbon with attached base spokes. Protein
surfaces are protein-only and are off by default in the interactive notebook.
The protected-category movie uses `55%` ChimeraX transparency for the protein
context surface. Active residues override that background with `8%`
transparency, matching cartoon colors, and thicker side-chain sticks. The
qualitative Coulombic movie uses an opaque protein
surface and one fixed `-10` to `+10 kcal/(mol e)` scale at `298 K`. All `1,007`
ColabFold models were accepted as preserved folds: `978` are in the strong-fold
review class and `29` are in the good-fold review class. There is no rejected
fold class to animate as a failure comparison. The proposal movie therefore
fits the `738` models retained by the local-geometry review to one cryo-EM
reference over the same `309` mapped C-alpha atoms. One centered stream cycles
the distal, peripheral, and combined chapters, and at most two atomic models
are open at once. Each candidate frame reports exact WT sequence identity over
the canonical `320`-residue RT and the corresponding substitution count.
The candidate layer shows all modeled side chains; the sticks are not a
mutation-only highlight.
Candidate dwell time is distributed evenly within each chapter; identity and
mutation count describe sequence change and do not rank candidate quality. The three
movies start `180 degrees` from the approved interactive pose and return to that
same offset pose after each full turn. The animation is qualitative, while local
RMSD remains the recorded structural measurement.

### Structure And Sequence Authority

- Structure authority is `ec86kit_7v9u_protomer1`: PDB `7V9U`, RT chain `A`,
  retained DNA chain `D`, and retained RNA chains `E/F`.
- The selected protomer is a modeling unit, not evidence that a designed
  RT-msDNA complex is monomeric.
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
tie-breakers. Exact F10/R13 substitutions and the Wang R13A evidence match are
reported annotations. Net charge, total mutation count, and inferred
oligomeric state do not rank rows. The current identities were determined by
mutation-position distance, with exact-substitution distance resolving one
distal pair tie; later evidence did not determine a selected identity.

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
  optimum or activity score. Wang places F10 and R13 at an RT-msDNA
  cross-protomer interface. R13A disrupted that interface while retaining
  msDNA and the tested antiphage phenotype. This supports exact interface-state
  reporting and a future fixed-R13A monomerization test. It does not establish
  that untested F10/R13 substitutions are monomerizing or harmless.
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

A panel required to share a monomerization intervention would need R13A fixed
in every complete generation policy, followed by fold review of those new
sequences. The existing pool cannot supply such a panel by selection alone.
Adding R13A after selection would create new protein sequences and would not
make the existing ProteinMPNN records monomer designs.

### Twist Handoff

The synthesis handoff contains all eight selected full-length CDS designs. Each
record is `963 bp`, including the WT stop
codon, and translates to the corresponding canonical `320 aa` fold input.
Unchanged amino acids retain the authoritative WT codon; changed amino acids use
the highest-frequency codon in the packaged E. coli table. This is
substitution-only minimal recoding, not whole-gene codon optimization.
Independent vendor codon optimization is disabled. The handoff hashes the
codon table with its other inputs, but the repository does not record the
table's external source. That citation should be resolved or the limitation
accepted before the method is frozen.

The bundle contains one upload CSV, one FASTA, and one annotated GenBank file
per candidate. GenBank variation features use compact amino-acid labels such as
`A47K`. The manifest reports global and 50-bp-window GC measurements, maximum
homopolymer length, repeated 20-mer count, internal BsaI/BsmBI checks, exact
F10/R13 states, the Wang R13A evidence match, and the unresolved RT-msDNA
assembly state for every ordered sequence.

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
