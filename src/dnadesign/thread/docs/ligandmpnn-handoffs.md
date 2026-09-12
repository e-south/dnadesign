---
doc_id: dnadesign-thread-ligandmpnn-handoffs
surface: tool-docs
owner: dnadesign-maintainers
last_verified: 2026-09-12
---

# LigandMPNN command and artifact handoffs

External studies send declarative JSON requests to Thread in its own installed
environment. Thread admits requests, verifies pinned parser context, builds
attested commands, and normalizes completed outputs. The study chooses residue
sets, seeds, comparisons, and acceptance policy; it reads the resulting JSON
without importing Thread or loading upstream Torch containers.

Run `python -m dnadesign.thread.adapters.ligandmpnn.cli --help` in the producer
environment. Each command reads one `--request` JSON file and emits one complete
JSON object on stdout only after successful validation. Capture that stream as
an artifact and check the exit status before publishing it. Errors go to stderr.
Unknown fields, duplicate keys, unsupported versions, and nonfinite JSON values
fail admission. There is no installation, checkout discovery, or model download.

| Command | Input | Result and side effects |
| --- | --- | --- |
| `identity` | None | Installed distribution/source identity and adapter source digest; no upstream execution |
| `preflight` | Upstream pin object | Exact checkout/checkpoint checks; failure exits nonzero |
| `context` | `thread.ligandmpnn.context_command_request` v1 | Runs the pinned parser and publishes its existing context inventory; stdout identifies its path and digest |
| `score-plan` | `thread.ligandmpnn.score_command_request` v1 | `thread.ligandmpnn.score_plan` v1 with canonical request digest and exact per-seed commands; no model execution |
| `design-plan` | `thread.ligandmpnn.design_command_request` v1 | `thread.ligandmpnn.design_plan` v1; materializes the requested alphabet sidecar and declares commands |
| `score-normalize` | The published score plan | Revalidates exact commands/completions and exports the result receipt plus all raw probabilities |

Planning and context commands require `--checkout-root` and `--execution-root`.
`preflight` requires only the checkout. Plan commands accept
`--python-executable` for the interpreter that will execute the pinned upstream
commands. The caller executes the returned argv arrays explicitly, with the
declared execution root as the working directory; plans do not execute models.
Design plans expose `residue_alphabet_sidecar` as the existing typed path/digest
receipt, or null when no alphabet was requested. Consumers can validate and
inventory those bytes without deriving an omission alphabet or decoding argv.

## Request fields

Every request carries its named `schema_id` and integer `schema_version: 1`.
The versioned fields below are explicit; changes to Python models do not extend
the JSON protocol implicitly.

Score and design requests contain `request_id`, `pdb_path`, `pdb_sha256`,
`output_dir`, `upstream`, `context_inventory`, `fixed_residues`,
`redesigned_residues`, `seeds`, `batch_size`, `number_of_batches`,
`use_atom_context`, and `use_side_chain_context`. Empty residue selectors are
arrays. A residue contains `chain_id`, integer `residue_number`, and
`insertion_code` (the empty string when absent). Both selector lists cannot be
nonempty. The paths are portable relative paths validated by the existing
adapter contracts.

Score requests additionally contain `mode` (`single_aa` or `autoregressive`)
and boolean `use_sequence`. Design requests instead contain `temperature`,
`residue_alphabets`, and `packing`. Each alphabet row contains `residue` and
`allowed_amino_acids`; packing contains `enabled`, `number_of_packs_per_design`,
`repack_everything`, and `use_ligand_context`.

Context requests contain `request_id`, `pdb_path`, `pdb_sha256`, `output_path`,
`upstream`, `minimum_nucleotide_atoms`, `required_polymer_types` (an array of
`dna`/`rna`), `chains`, `parse_all_atoms`, and
`parse_atoms_with_zero_occupancy`. They retain the existing pinned-parser
admission rules.

All upstream pin objects contain `commit`, `checkpoint_path`,
`checkpoint_sha256`, `packing_checkpoint_path`, and
`packing_checkpoint_sha256` (null when packing is not requested). Pin hashes
and `pdb_sha256` are bare hexadecimal digests; `context_inventory` is the
existing `{path, sha256}` reference with a `sha256:` URI.

## Staged score planning

`score-plan --input-root STAGING --execution-root FINAL` validates the input
PDB and context inventory in staging while binding commands to the final
execution directory. The pinned checkout must be absolute and outside staging.
Explicit interpreter paths must also remain outside staging. A symlink route
through the moving directory is rejected even when its target is external.
After the caller promotes the unchanged input tree, ordinary planning at the
final root must reproduce the same commands. Changed input bytes still fail
validation. Omitting `--input-root` retains the existing requirement that
execution inputs already exist at the execution root.

## Reading completed scores

`score-normalize --request score-plan.json --execution-root RUN
--trust pinned_local_execution` accepts only the explicitly trusted local
outputs of the planned commands. It uses the existing restricted Torch loader,
context replay, command digest, completion-record, seed, shape, and alphabet
checks. It does not turn downloaded or otherwise untrusted `.pt` files into
safe inputs. The caller's trust attestation is required even when bytes have
matching digests.

The output is `thread.ligandmpnn.score_export` v1. `result` preserves the
existing `thread.ligandmpnn.score_result` v3 receipt. `plan_sha256` binds the
canonical plan JSON. Each `probability_artifacts` row contains `seed`,
`output_sha256`, `residue_names`, `shape`, `dtype`, `values`, and `sha256`.
`values` is a numeric JSON array with shape `[draw, residue, 21]`; its last
state is `X`. It retains the complete raw probability values without
conditioning, rounding, residue selection, or summary substitution.

Each artifact digest is SHA256 over that row with only its `sha256` field
removed, encoded as UTF-8 JSON with sorted keys, separators `,` and `:`,
ASCII-escaped non-ASCII characters, and nonfinite values prohibited. A consumer validates the schema, digests, exact
seed/output accounting, dtype, shape, and result lineage before applying its
own policy. Numeric JSON uses more space than a binary tensor; it provides a
single standard, pickle-free artifact suitable for these bounded probes.

Retained exports can be read without the producer environment. Their
provenance records past producer validation; artifact reading alone is not a
fresh upstream parser replay or new model execution. Preserve the exported
bytes and their accepted lineage when moving them between systems.

Return to [Thread](README.md) for adapter guarantees and ownership boundaries.
