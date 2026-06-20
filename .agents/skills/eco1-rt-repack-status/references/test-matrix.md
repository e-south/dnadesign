# Test Matrix

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | Where is eco1_rt_repack now? | Use the checked-in study record and report record-only status. | Pass if the answer reports phase, blockers, and next route without inventing an OPS provider. |
| Trigger negative | Where is rt_lnrna_sponging_construct_triage now? | Route away from this study-specific skill. | Pass if the skill does not generalize Eco1 RT repack status to the RT-lnRNA study. |
| Contract boundary | Is `thread` implemented? | Say no; route to the dev spec and fixture scaffold. | Pass if the answer distinguishes planned contracts from source code. |
| Policy routing | Which residues are fixed? | Route to residue-mask policy and conservative-mask fixture. | Pass if catalytic/contact/conservation masks stay study-owned. |
| Gate routing | What blocks MPNN request generation? | Route to structure authority, mask contract, and sampling plan gates. | Pass if the answer does not jump to backend execution. |
| Conservation routing | What blocks conservation masking? | Route to the MSA method, conservation source contract, source-sequence materializer plus sufficiency gate, `aligner.msa` for generic aligned FASTA bundles, and the conservation materializer. | Pass if the answer says real provider caches/sufficiency-passing source FASTA/aligned FASTA inputs are still missing and does not imply live fetch/provider fallback or conservation scoring inside aligner. |
| Handoff routing | What blocks RT-lnRNA use of these candidates? | Route to the downstream handoff readiness check. | Pass if paired construct semantics stay downstream. |
| Repeatability | Run the skill audit twice. | Structural and routing checks remain deterministic. | Pass if both runs finish with no failures and no generated outputs are required. |
| Source/test IA regression | Add a flat operation module, flat primitive materializer, or flat materialization test. | Audit fails before the status skill can be treated as healthy. | Pass if only semantic `operations/contracts/`, `operations/materialization/<primitive>/`, `tests/contracts/`, and `tests/materialization/<primitive>/` packages are accepted. |
