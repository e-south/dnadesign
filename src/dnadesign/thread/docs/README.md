---
doc_id: dnadesign-thread-docs
surface: tool-docs
owner: dnadesign-maintainers
last_verified: 2026-08-14
---

# Thread

Thread standardizes the files passed between protein-design and
structure-analysis tools. Its contracts keep sequence identity, runtime
provenance, results, and checks explicit. A caller supplies the scientific
selection and decides what the results mean.

## Routes

| Job | Surface | Result |
| --- | --- | --- |
| Prepare a ProteinMPNN run | `dnadesign.thread.adapters.proteinmpnn` | Validated request sidecars and a request manifest |
| Declare a pinned LigandMPNN run | `dnadesign.thread.adapters.ligandmpnn` | Preflight report, digest-bound alphabet sidecars, explicit commands, and a planned receipt |
| Declare a LigandMPNN probability probe | `dnadesign.thread.adapters.ligandmpnn` | Explicit single-AA or autoregressive `score.py` commands |
| Read ProteinMPNN samples | `dnadesign.thread.adapters.proteinmpnn` | Normalized backend sample rows |
| Build stable candidate rows | `dnadesign.thread.candidates` | Sequence, mutation, and mask-audit fields |
| Read ColabFold output | `dnadesign.thread.adapters.colabfold` | Normalized confidence, PAE, and RMSD rows |
| Define a fold check | `dnadesign.thread.foldcheck` | Backend-neutral request and report artifacts |
| Record predicted structures | `dnadesign.thread.structure_predictions` | Provenance-separated registry rows |
| Read Atlas annotations | `dnadesign.thread.adapters.esm_atlas` | Sparse protein and residue annotations |
| Query Biohub ESMC | `dnadesign.thread.adapters.biohub_esmc` | Authenticated sparse annotations with redacted runtime metadata |
| Inspect existing structures | `dnadesign.thread.structure_views` | An interactive browser view |

## Boundaries

The LigandMPNN adapter always declares `--model_type ligand_mpnn` and keeps
fixed or redesigned residues, atom and fixed-side-chain context, packing,
seeds, temperature, and sample counts explicit. Residue-specific allowed
alphabets contain only the 20 canonical amino acids and are translated into a
deterministic, SHA256-bound official `--omit_AA_per_residue` JSON sidecar; a
bare sidecar path is not accepted. The upstream noncanonical `X` state is
always omitted. Atomic planners may write the bytes to a staging path while
binding a distinct final execution path; only the final path is serialized,
and the promoted file has an explicit digest-validation method.

Probability probes use the official `score.py` single-AA or autoregressive
mode with explicit sequence, atom-context, and fixed-side-chain-context flags.
The request requires at least 10 batches because the pinned upstream recommends
that minimum for decoding-order-dependent probabilities. This is an execution
stability policy, not a universal biological or statistical threshold.

The CLI contracts were checked
against official upstream commit
`26ec57ac976ade5379920dbd43c7f97a91cf82de`. The caller supplies the pinned
checkout and checkpoint hashes; the adapter does not clone, download, execute,
choose residues, or interpret designs.

- Adapters own translation to or from one external tool. They do not own
  candidate selection or biological interpretation.
- `foldcheck` owns artifact shape and explicit thresholds supplied by the
  caller. It does not choose a folding backend or acceptance policy.
- `structure_predictions` keeps each backend result as a separate provenance
  row. Results from different runtimes are never merged into one authority.
- `structure_views` renders existing PDB or mmCIF content for interactive
  notebook inspection. Its py3Dmol HTML is a browser viewer, not a scientific
  plot, evidence bundle, or replacement for a deterministic BaseRender figure.
- Model execution, credentials, schedulers, and device storage remain explicit
  operator concerns. There is no hidden fallback.

Study-owned code may call these public packages after it has chosen subjects,
masks, thresholds, and comparison policy. Thread must not import study code,
name study objectives, or publish promotion decisions.
