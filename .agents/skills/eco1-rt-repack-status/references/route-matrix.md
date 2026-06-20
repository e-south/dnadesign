# Route Matrix

Use this matrix when a question is near Eco1 RT repack status but may belong to
another owner surface.

| User question | Primary surface | Why |
| --- | --- | --- |
| Where is `eco1_rt_repack` now? | `docs/studies/eco1_rt_repack/record/status.md` plus `operations/ops.study.yaml` | The study is record-only. No OPS provider is registered. |
| What is the higher-order development plan? | `docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md` | The dev spec owns generic `thread` boundaries and tracer-bullet sequencing. |
| What code should be implemented first? | `contexts/implementation-roadmap.md` | The roadmap owns code homes, artifact order, and negative-path slice boundaries. |
| Which residue positions are protected? | `contexts/residue-mask-policy.md` plus `operations/contract/fixtures/thread/conservative_mask_cases.yaml` | Protection policy is Eco1 profile-specific until residue maps exist. |
| How will MPNN candidates be validated? | `contexts/fold-validation-policy.md` | Fold-check thresholds and no-go signals are policy, not execution. |
| Can this become an oligo-pool or multipart design? | `contexts/synthesis-feasibility-policy.md` | Synthesis feasibility is a computational handoff decision after candidate localization is known. |
| What files define the planned artifacts? | `operations/contract/surfaces/artifacts.yaml` plus `operations/contract/schemas/thread-artifact-chain.schema.yaml` | Artifact names stay generic while Eco1 is a profile id. |
| Where should validator edits go? | `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/` | The CLI stays thin; semantic checks are split by profile, source authority, structure artifacts, evidence artifacts, and mask cases. |
| Where should materializer edits go? | `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/<primitive>/` | Materializers are grouped by runtime primitive package; do not add flat `operations/materialization/*.py` or `operations/*_materialization.py` modules. |
| Where should tests go? | `src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/` or `src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/<primitive>/` | Tests mirror source ownership and avoid a flat study or materialization test root. |
| What blocks structure or residue mapping? | `operations/contract/readiness/checks/structure_authority.yaml`, `workbench/provenance/residue-numbering-policy.yaml`, and `outputs/thread/eco1_rt_conservative_v1/` | Structure authority, numbering policy, and local structure artifacts are materialized. |
| What blocks contact masking? | `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/` and `outputs/thread/eco1_rt_conservative_v1/contact_profile.parquet` | Contact evidence is materialized locally from retained DNA/RNA context distances. |
| What blocks conservation masking? | `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/`, `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/`, `src/dnadesign/aligner/msa/`, and `contexts/msa-method.md` | Source-sequence, source-bundle sufficiency, conservation-profile, and generic MAFFT bundle seams exist, but real provider caches, sufficiency-passing source FASTA bundles, aligned FASTA inputs, and conservation profile are still missing. |
| What blocks MPNN request generation? | `operations/contract/readiness/checks/mask_contract.yaml` and `sampling_plan.yaml` | Sampling requires materialized conservation evidence, mapped masks, and explicit backend policy. |
| What blocks the first candidate handoff? | `operations/contract/readiness/checks/candidate_handoff.yaml`, `foldcheck_runtime.yaml`, and `assembly_feasibility.yaml` | Candidate handoff depends on QA and feasibility, not just selected ids. |
| What blocks RT-lnRNA collaboration? | `operations/contract/readiness/checks/downstream_rt_lnrna_handoff.yaml` and `operations/contract/schemas/rt-lnrna-candidate-acceptance.schema.yaml` | Collaboration is downstream RT-only acceptance, not shared construct ownership. |
| Should this use `permuter`? | Dev spec owner-boundary table | `permuter` owns DMS and explicit variant intent, not inverse-folding proposal generation. |
| Should this use `aligner`? | `src/dnadesign/aligner/msa/` for aligned FASTA bundles; Eco1 source-sequence bundles remain in the study materializer. | `aligner` can run generic MAFFT bundle mechanics, but Eco1 source authority, target-row policy, provider-cache bundling, conservation scoring, and masks stay study/thread-owned. |
| Is `thread` implemented? | `src/dnadesign/studies/units/eco1_rt_repack/README.md` and the dev spec | The current scaffold is docs and fixtures only. |

Routing boundary:

- Use this skill for Eco1 RT repack status and policy routing.
- Use `rt_lnrna_sponging_construct_triage` surfaces only after the question is
  about paired construct subjects or downstream promotion.
- Use future `thread` docs only after executable contracts are created.
