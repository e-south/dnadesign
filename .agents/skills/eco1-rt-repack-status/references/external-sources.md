# External Sources

External papers and structures explain method choice. The checked-in Eco1
record remains the source of truth for current artifacts, hashes, and gates.

## Source Role Table

| Source | Retrieved | Study role | Must not be used as |
| --- | --- | --- | --- |
| Tao et al. 2026, Nature Biotechnology, DOI `10.1038/s41587-026-03149-6` | 2026-06-24 | Method prior for fixed-backbone RT redesign, protected functional residues, MSA-derived WT plurality/frequency masking, and fold-check filtering. | A prime-editing objective for this study or a mandate to use a whole-database retron MSA. |
| `Allentaoyz/Redesigned_prime_editor_RTs` public repository | 2026-06-23 | Implementation-context prior for Tao-style redesign workflow shape. | A direct Eco1 source authority or a runtime dependency. |
| `dauparas/ProteinMPNN` public repository | 2026-06-25 | Backend request-format and execution prior for helper-compatible parsed-PDB JSONL, assigned-chain JSONL, fixed-position JSONL, chain-local 1-indexed positions, explicit designed chain selection, `protein_mpnn_run.py`, sampling temperature, seed, `num_seq_per_target`, and omitted-amino-acid flags. | A license to infer Eco1 mask policy, run a backend without a request manifest, or use raw PDB residue numbering as fixed-position numbering. |
| Mestre et al. 2020, Nucleic Acids Research, DOI `10.1093/nar/gkaa1149` | 2026-06-23 | Retron RT roster and classification authority. Mestre S1 defines the candidate pool, Ec86 RT clade 9 panel, and II-A3/`42_1` family panel. | A finished MSA, conservation profile, or full-roster Phase 1 denominator. |
| Simon et al. 2019, Nucleic Acids Research, DOI `10.1093/nar/gkz865` | 2026-06-23 | RT motif and figure-grammar prior for RT1-RT7, Region X/Y, NAxxH, catalytic DD/YADD, and VTG annotations. | A mask-authority table by itself; Eco1 canonical coordinates must come from the checked-in ontology. |
| Wang et al. 2022, Nature Microbiology, DOI `10.1038/s41564-022-01197-7` | 2026-06-23 | Eco1/Ec86 cryo-EM structure prior for RT-msDNA/msrRNA context, active-site/motif context, RT1-RT7 interval spans, and interface-candidate residues. | Automatic permission to promote every interface candidate or to redesign substrate-contact residues. |
| RCSB PDB `7V9U` | 2026-06-24 | Structure cross-check for the selected Ec86 RT-msDNA-RNA complex authority. | Replacement for the study-owned ec86kit protomer authority or residue-numbering policy. |
| [Candido et al. 2026, `Language Modeling Materializes a World Model of Protein Biology`](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1) | 2026-06-25 | Representation prior for ESMC, ESMFold2, ESM Atlas, and sparse autoencoder feature interpretation. | Experimental evidence for processivity, strand displacement, hairpin readthrough, or Eco1 candidate acceptance. |
| ESM Atlas API documentation | 2026-06-25 | Public API behavior prior for Atlas protein lookup, feature summaries, sparse activations, alpha-status caveats, and no-auth smoke tests. | A stable production API guarantee or a replacement for ColabFold/AlphaFold-family fold validation. |
| [Biohub `/api/v1/encode` and `/api/v1/logits` documentation](https://www.biohub.ai/api-reference/logits) | 2026-06-26 | Authenticated public API behavior prior for ESMC tokenization, sequence logits, and query-time SAE activations on synthetic sequences. | A fold-validation endpoint, Atlas database insertion path, processivity model, or candidate acceptance gate. |
| [Biohub ESMC 6B SAE model card](https://huggingface.co/biohub/ESMC-6B-sae-layer60-k64-codebook16384) | 2026-06-29 | Source for the current Eco1 query-time SAE dictionary name, layer, top-k sparsity, codebook size, and source-backed feature-description scope. | Curated Eco1 function labels, assay evidence, or permission to treat model-derived feature descriptions as processivity measurements. |
| [Biohub ESMC SAE feature-interpretation notebook](https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb) | 2026-06-29 | Reference workflow for inspecting top SAE features, per-residue activations, and source-backed feature descriptions for the exact queried SAE dictionary. | Permission to mix Atlas labels into a different SAE dictionary, a functional annotation source, or evidence for processivity, strand displacement, or candidate acceptance. |
| [Biohub ESMC mutation-scoring notebook](https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/esmc_mutation_scoring.ipynb) | 2026-06-27 | Reference implementation pattern for masked sequence-logit inference: mask one residue with `_`, compute per-position entropy, compute zero-shot single-substitution LLRs relative to the reference residue, and plot entropy, negative-LLR fraction, and LLR heatmap. | Experimental DMS data, an Eco1 mask rule, or evidence for RT activity/processivity. |

## Placement Rules

- MSA source authority lives in
  `docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml`.
- MSA reproduction and reviewer-facing rationale live in
  `docs/studies/eco1_rt_repack/contexts/msa-method.md`.
- Mask-authoritative motif anchors and RT intervals live in
  `docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml`.
- Visualization-only RT annotation tracks live in
  `docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml`.

For live factual claims, re-check external sources and cite them in the user
response. The checked-in study record should keep concise source identifiers,
stable provenance fields, and explicit no-inference boundaries.
