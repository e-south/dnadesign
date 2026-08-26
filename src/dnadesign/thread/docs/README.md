---
doc_id: dnadesign-thread-docs
surface: tool-docs
owner: dnadesign-maintainers
last_verified: 2026-08-26
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
| Inventory LigandMPNN atom context | `dnadesign.thread.adapters.ligandmpnn` | Pinned-parser request, command, and digest-bound observed-atom receipt |
| Declare a pinned LigandMPNN run | `dnadesign.thread.adapters.ligandmpnn` | Preflight report, digest-bound context and alphabet references, explicit commands, and a planned receipt |
| Run and read a LigandMPNN probability probe | `dnadesign.thread.adapters.ligandmpnn` | Explicit `score.py` commands and a context-proven executed-result receipt |
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

Generated design and score commands execute from a temporary source tree
materialized directly from the pinned Git commit. They do not import helper
modules or bytecode caches from the mutable checkout. Immediately before each
run, the wrapper verifies the declared checkpoint digests, copies those exact
bytes into the isolated runtime, and points the upstream entrypoint at the
verified copies. The input PDB and optional residue-alphabet sidecar are
verified and staged the same way, so execution cannot substitute changed input
bytes after planning. The wrapper rejects abbreviated, duplicate, multi-input,
or alternate-model options that could redirect upstream parsing around those
staged files. Preflight remains an early diagnostic; execution repeats
the source, input, and weight identity boundary rather than trusting an earlier
check.

An atom-context request is not evidence that context was parsed. Before a
design or probability request is admitted, the context probe imports
`data_utils.parse_PDB` from the exact clean upstream commit and records the
effective nonprotein atoms returned in upstream `Y`, `Y_t`, and `Y_m`. The
probe associates those effective rows back to the same upstream `other_atoms`
selection to retain atom name, element, chain, residue name, residue number,
and insertion code. It fails before writing a receipt when no expected DNA or
RNA atoms survive the upstream parser. Standard PDB DNA residue names
`DA/DC/DG/DI/DT/DU` and RNA residue names `A/C/G/I/U` (plus their `R`-prefixed
forms) are classified; any other residue identity remains explicitly `other`.
This classification labels the observed atoms—it does not decide which atoms
upstream consumes.

Context receipts are published by an atomic no-follow replacement. If the
post-replacement directory durability check fails, publication restores the
prior regular-file bytes or prior absence before reporting ordinary failure.
Every newly created output-directory entry is synced in its parent before the
receipt can be reported durable. Public receipt loading opens every ancestor
and the leaf descriptor-relatively with no-follow flags, verifies the opened
leaf is regular, and hashes the same single byte read that is decoded.
An existing symlink, FIFO, or other nonregular receipt is rejected before
replacement because it cannot be restored as regular receipt content.
If that restoration cannot itself be made durable, the public
`LigandMpnnContextPublicationUncertainError` reports the uncertain state.
Admission replays the exact pinned parser over the digest-matched PDB and
requires the loaded receipt to equal that derived inventory; a caller-authored
JSON receipt cannot create observed context by assertion alone.

Score execution completion records use the same lexical, descriptor-relative
no-follow loading boundary. Runtime publication syncs the completion file and
its parent directory. If completion durability fails after score publication,
the runtime removes and syncs both lifecycle entries before reporting ordinary
failure; an unsuccessful rollback raises the public
`LigandMpnnCompletionPublicationUncertainError` instead.
Each newly created score-output directory link is likewise created through the
no-follow descriptor chain and synced in its receiving parent. Score hard-link
publication syncs the destination directory; if that sync fails, the runtime
removes and syncs the score before reporting ordinary failure. An unsuccessful
score rollback raises `LigandMpnnScorePublicationUncertainError`, and no
completion record is published for either failure state.

Probability probes use the official `score.py` single-AA or autoregressive
mode with explicit sequence, atom-context, and fixed-side-chain-context flags.
The request requires at least 10 batches because the pinned upstream recommends
that minimum for decoding-order-dependent probabilities. This is an execution
stability policy, not a universal biological or statistical threshold.

Each score request binds the input PDB digest and an immutable context-inventory
receipt before execution. Result parsing requires that receipt to match the
same PDB, upstream commit, all-chain parser scope, side-chain parsing mode, and
positive-occupancy default. PDB request paths are safe, non-option relative paths; an
option-looking output directory is rejected before command construction.
Each generated command binds every parsed wrapper field and the complete
upstream argument vector to a canonical digest. Successful execution emits an
exclusive per-seed completion record containing the actual parsed arguments,
and result admission requires that record to equal the planned command before
accepting artifacts. Standalone abbreviations, changed values, simultaneous
fixed/redesigned selectors, and unique unplanned overrides therefore fail
closed. Score execution uses a uniquely owned temporary output directory and
atomically publishes the final `.pt` without replacement, so concurrent
requests cannot overwrite a shared basename. Result parsing then requires
exactly one upstream `.pt` artifact per requested seed. The executed receipt
records the semantic request, input, command-set, execution, per-command, checkpoint,
upstream-commit, output identities, parser source digest, and full observed
context inventory. `atom_context_status` can become
`enabled_with_observed_nucleotide_context` only after those checks pass; it is
never copied directly from the requested flag. Missing, extra, symlinked,
schema-drifted, zero-nucleotide-context, wrong-seed, wrong-alphabet, or
wrong-shaped outputs fail closed.

The pinned upstream has one mode-specific shape distinction: single-AA results
carry a `[draw, residue, decoding-position]` decoding-order tensor, while
autoregressive results carry `[draw, decoding-position]`. The parser validates
each as complete residue permutations and never coerces one shape into the
other.

Official `score.py` stores NumPy arrays in a PyTorch pickle-capable container.
The parser therefore requires an explicit `pinned_local_execution` trust
attestation and restricts `torch.load` with `weights_only=True` plus only the
NumPy globals required by the pinned output schema. These controls are defense
in depth, not a sanitizer: do not parse downloaded, emailed, or otherwise
untrusted `.pt` files. The parser reads each artifact's bytes once, hashes those
exact bytes, then parses the same in-memory bytes to avoid a hash/load race.

The raw 21-state alphabet is required to be
`ACDEFGHIKLMNPQRSTVWYX`. Results retain the complete raw probabilities and the
raw `pX` values. Thread never silently drops `X`: a canonical-20 view is exposed
only when the caller passes a `LigandMpnnCanonical20Policy`, which performs
explicit conditioning on a canonical residue and applies only the numerical
minimum-mass guard chosen by that caller. Any biological `pX` eligibility rule,
context-comparison threshold, residue subset, or promotion decision remains
study-owned.

The CLI contracts were checked
against official upstream commit
`26ec57ac976ade5379920dbd43c7f97a91cf82de`; its `data_utils.py` digest at that
commit is
`sha256:a39c4674977786f9fa697f962d4e91ec79989ede30a79ad389d698291d0484c8`.
The caller supplies the pinned checkout and checkpoint hashes. The adapter does
not clone, download, choose residues, or interpret designs; only the explicit
context-probe entrypoint executes upstream parsing.

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
