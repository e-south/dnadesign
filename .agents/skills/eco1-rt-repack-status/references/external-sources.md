# External Sources

External paper and structure references are contextual, not the checked-in
record of truth.

Current method source:

- Tao et al., "AI-guided redesign of laboratory-evolved reverse transcriptases
  enhances prime editing", Nature Biotechnology, published 2026-05-21, DOI
  `10.1038/s41587-026-03149-6`.
- The associated public repository is `Allentaoyz/Redesigned_prime_editor_RTs`.

Use these references to understand the computational pattern: fixed-backbone
sequence design, residue protection, candidate sampling, and fold-check
filtering. Do not import the paper's prime-editing objective into the Eco1 RT
sponging study.

Current conservation-source priors:

- Mestre et al., "Systematic prediction of genes functionally associated with
  bacterial retrons", Nucleic Acids Research, 2020, DOI
  `10.1093/nar/gkaa1149`.
- Mestre Supplementary Table S1 is the current accession-roster prior and
  candidate pool for `broad_tao_homolog_rt` plus the clade-near source for
  `eco1_like_retron_rt`; it is not itself a materialized MSA, conservation
  profile, or broad conservation denominator.
- The selected MSA source policy is recorded in
  `docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml`.
- The future-agent reproduction method is recorded in
  `docs/studies/eco1_rt_repack/contexts/msa-method.md`.
- Simon et al. 2019, Wang et al. 2022, and Khan et al. 2025 are secondary
  sanity/context sources for motifs, structure, and functional retron rosters.

For live factual claims, re-check external sources and cite them in the user
response. The checked-in study record should keep only concise source
identifiers and stable provenance fields.
