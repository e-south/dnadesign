## Eco1 RT Repack Command Groups

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-22

This directory is the reproducibility route for the study-owned Eco1 RT repack
materialization path. It is not a hidden run-all pipeline. Each lane in
`pipeline.yaml` names one owner, one artifact boundary, and one executable
command or planned future command.

Use `pipeline.yaml` as the machine-readable checklist for rerunning the current
Phase 1 artifact chain:

```text
structure authority
structure preprocessing
contact evidence
contact geometry evidence
provider source cache
roster source cache
source FASTA bundles
source sufficiency
alignment bundles
MSA visualization sidecars
conservation profile
manual mask authority
mask set
contact-risk audit
phase validation
```

### Rerun Ladder

Use this order when regenerating the local Phase 1 state. Each step maps to a
lane id in `pipeline.yaml`; read that lane for the exact command and external
input placeholders.

1. `structure_authority` emits `backbone_bundle.yaml` and `residue_map.parquet`.
2. `structure_preprocessing` emits `structure_preprocessing_manifest.yaml` to
   make the raw 7V9U dimer to selected protomer-1 chain ontology explicit.
3. `contact_profile` emits retained DNA/RNA nearest-proximity evidence.
4. `contact_geometry_profile` emits atom-class side-chain/backbone/contact-density
   evidence from the selected protomer mmCIF.
5. `conservation_provider_sources` ingests the hash-pinned Mestre S1 roster and
   explicit provider source files or writes unresolved-provider ledgers.
6. `conservation_roster_cache` emits selected Ec86 clade 9 and II-A3/`42_1`
   source records with declared QC metadata.
7. `conservation_source_bundles` emits unaligned FASTA bundles and inserts the
   ec86kit target row.
8. `conservation_source_sufficiency` must pass before any MSA backend runs.
9. `conservation_alignments` runs the declared Clustal Omega backend through
   `dnadesign.aligner.msa`.
10. `evidence_profiles` emits `conservation_profile.parquet`.
11. `manual_mask_authority` emits the runtime manual mask-authority artifact
   from the checked-in ontology.
12. `mask_contract` emits the simple clade9-plurality-25/direct-contact-5 A
   `mask_set.yaml`.
13. `contact_risk_profile` emits a contact evidence review from the contact,
   conservation, manual-mask, and mask-set evidence chain.
14. `surface_accessibility_profile` emits complex-context SASA evidence from
   the selected Ec86 RT-msDNA-msrRNA structure. This is an earlier check, not
   an input to the current mask rule.
15. `phase1_contract_validation` must pass before sampling-plan work starts.

Phase 1 validation is not a presence-only check. It validates
`structure_preprocessing_manifest.yaml` as the raw 7V9U-to-protomer authority and
re-checks `contact_geometry_profile.parquet` upstream hashes against the current
structure-source policy, preprocessing manifest, backbone bundle, residue map,
and ec86kit model.

The contact-geometry implementation is split by responsibility so this evidence
surface remains easy to audit before mask work: `structure_io.py` owns
mmCIF parsing and retained-chain extraction, `rows.py` owns atom-distance and
contact-density row construction, `writer.py` owns Parquet schema/metadata
emission, and `pipeline.py` owns orchestration only.

The Phase 1 state is materialized locally through
`contact_geometry_profile.parquet`, `contact_risk_profile.yaml`, and
`mask_set.yaml`. Evidence-review artifacts do not decide protected residues.
The current mask rule is
`eco1_rt_clade9_plurality25_direct_contact5a_v1`: protect NAxxH/YADD/VTG,
Wang/Ec86 direct substrate-contact priors, Ec86 clade 9 >=25% WT-plurality
conservation calls, and mapped residues within 5 A of retained DNA/RNA.
Terminal residues `1`, `2`, and `312-320` are `non_fixed_missing_backbone`.

### Next Policy Gate

The next execution step is a sampling request plan:

```text
thread_plan.yaml
```

The next slice should materialize a small explicit `thread_plan.yaml` from the
validated simple mask, with backend, seed, temperature, request hash, fixed/
non-fixed position source, and no-fallback policy. Any future mask expansion
must be opened as an explicit policy change before it can feed sampling.

### Source-Role Guardrails

- Tao is the masking-method prior: homolog MSA conservation, fixed functional
  residues, fixed-backbone RT redesign, and fold-check triage.
- Mestre is the source ontology: the full S1 roster is a candidate/context
  surface, while Ec86 RT clade 9 and II-A3/`42_1` are the active conservation
  denominators.
- Simon is the annotation grammar for RT regions and motif visualization.
- Wang is the Eco1/Ec86 structural prior for the selected cryo-EM context,
  active-site/motif spans, and candidate interface residues.
- Paired-protomer dimerization is not a retention objective for the current
  monomeric RT-msDNA-msrRNA design profile; alpha-1/pre-RT1 residues are not
  fixed solely for dimer preservation.
- `manual-mask-authority.yaml` is the source for NAxxH, YADD, VTG, RT1-RT7
  review labels, and Wang/Ec86 direct contact priors. Under the selected mask
  policy, NAxxH/YADD/VTG and Wang direct contacts are protected; RT1-RT7 labels
  do not blanket hard-fix residues.

Keep these lanes independently runnable when executable commands are
introduced. A future orchestration command may call them in order only after
each lane has its own validator and negative-path fixture. Do not collapse them
into a single hidden pipeline.

External data inputs remain explicit. The provider and roster cache lanes need
a hash-pinned Mestre S1 table plus explicit provider FASTA source roots. The
checked-in study record should never infer source rows from review figures,
public Eco1 accessions that disagree with the ec86kit target hash, or transient
local FASTA files.
