---
doc_id: junction-inspect-and-verify
title: Inspect and verify a junction bundle
type: guide
audience: reviewers of an existing junction design
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Inspect and verify a bundle

Verify the bundle first, then open only the artifact that answers the review
question.

## Verify exact contents

```bash
uv run junction verify <bundle-directory> --format json
```

Verification checks the exact inventory and hashes, parses the saved request,
reruns the recorded algorithm, regenerates each expected artifact, and rejects
any byte that no longer matches. Missing, extra, moved, malformed, or symlinked
entries fail.

## Find the evidence

| Question | Artifact |
| --- | --- |
| What exact request was used? | `request.json` |
| Which loci, toeholds, barcodes, and fragments were selected? | `plan.json` |
| Which seeds, budgets, scores, and evaluation counts were used? | `plan.json` → `assembly_groups[].search` |
| Which checks passed or were not run? | `checks.json` |
| Which complete oligos and primers are handed off? | `orders/oligos.tsv` |
| Which sequences can another tool consume? | `sequences/*.fasta` |
| Which typed records drive molecular review figures? | `views/*.json` |
| Have the files changed? | `manifest.json` plus `junction verify` |

`thermodynamic_screening` remains `not_run`. String distance cannot change
that state to `passed`.

## Trace one target

In `plan.json`, read the target's records in this order:

1. `fragments` for complete strands, roles, and target spans;
2. `junctions` for selected `t/t*`, `b/b*`, coordinates, and adjacent
   fragments;
3. `recovery` for supplied primers and expected terminal extensions;
4. `reconstructed_target` and `reconstructed_complement` for exact string
   identities; and
5. target-scoped checks for reconstruction, primer matches, and order lengths.

The plan retains selected choices and bounded search receipts, not every
rejected candidate.

## Draw only the view you need

BaseRender consumes `views/three_way_junction_review.v1.json` and writes a
separate create-only figure bundle. It does not change or verify the Junction
bundle.

| Review question | Renderer |
| --- | --- |
| Do the two oligos in each fragment align as intended? | `junction_annealed_fragments` |
| Where do the fragment pairs and interfaces sit across the target? | `junction_three_way_assembly` with `view: assembly` |
| Does each interface have the expected three-arm sequence geometry? | `junction_three_way_assembly` with `view: junction_detail` |

Use the [BaseRender Junction route](../../../baserender/docs/integrations/junction.md)
for job fields and selection limits. The [gene-scale
tutorial](../getting-started.md) shows all three views once.

Junction separately owns the sequence-dissimilarity record and plot because
that view computes Junction's search metrics. It compares selected toeholds,
barcodes, and combined strings; it is not a thermodynamic score.

## Review before ordering

The order table is vendor-neutral. Before procurement or experimental work,
review synthesis constraints, secondary structure, crosstalk, end preparation,
primer behavior, reaction conditions, downstream cloning, and experimental
controls in their owning workflows.

For the exact bundle layout, API, and failure contract, see [Artifacts, API,
and errors](../reference/artifacts-api-and-errors.md).
