---
doc_id: junction-getting-started
title: Build a gene-scale junction design
type: tutorial
audience: first-time users
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-11
---

# Build a gene-scale design

This tutorial starts with one 705 bp DNA sequence and two 20 nt primer-binding
sites, one at each end. `junction` divides the sequence into 13 two-strand
fragments, chooses 12 temporary three-way interfaces, writes the order files,
and confirms that the planned fragments reconstruct the submitted sequence.

The figures are optional checks. The order table and FASTA files are the
handoff artifacts.

## Before you start

Run the tutorial from the root of a cloned `dnadesign` repository. If the
environment is not installed yet, follow the repository [installation
guide](../../../../docs/setup/installation.md), then run:

```bash
# Install the exact dependency versions recorded by dnadesign.
uv sync --locked
```

## 1. Copy the example into a temporary directory

The target could also arrive as raw DNA, a text file, or FASTA. A FASTA record
looks like this; the sequence is shortened only in this display:

```fasta
>gene-scale-target
TCATTCTAGGCTAGCTGAGT...TATAGGGTTCTGCAGTACTCGGTCCTG
```

Prepare the checked-in example:

```bash
# Make a new temporary directory so generated files do not mix with source files.
JUNCTION_DEMO_ROOT="$(mktemp -d)"

# Resolve the directory to one physical path and make it available to later commands.
JUNCTION_DEMO_ROOT="$(cd "$JUNCTION_DEMO_ROOT" && pwd -P)"
export JUNCTION_DEMO_ROOT

# Copy the example request and the three optional drawing jobs.
cp src/dnadesign/junction/examples/gene-scale/request.yaml \
  "$JUNCTION_DEMO_ROOT/request.yaml"
cp src/dnadesign/junction/examples/gene-scale/jobs/*.yaml \
  "$JUNCTION_DEMO_ROOT/"
```

The request contains the target sequence, the two primer sequences, acceptable
fragment lengths, search limits, and a fixed random seed. Read those values
before substituting a real sequence.

You can also replace the example target with one or more FASTA records while
keeping the same planning limits:

```bash
# Read the planning limits from the example and the target sequences from FASTA.
uv run junction request \
  --base-request "$JUNCTION_DEMO_ROOT/request.yaml" \
  --input targets.fasta \
  --assembly-group assembly-01 \
  --primer-binding-length 20 \
  > "$JUNCTION_DEMO_ROOT/request-from-fasta.json"
```

FASTA IDs become target IDs. Malformed FASTA, duplicate IDs, ambiguity codes,
and invalid terminal primer matches fail before search begins.

## 2. Build the bundle

```bash
# Plan the oligos, publish a new bundle, and replay it once after publication.
uv run junction build "$JUNCTION_DEMO_ROOT/request.yaml" \
  --output "$JUNCTION_DEMO_ROOT/design-bundle" \
  --format json
```

`junction` chooses short target-derived toeholds, assigns temporary barcode
sequences, builds both strands of every fragment, and checks the final sequence
and every order length. The output directory must be new; the command will not
silently replace an earlier design.

## 3. Check the planned fragment pairs

```bash
# Render the two orderable strands for every planned fragment.
uv run baserender job run \
  "$JUNCTION_DEMO_ROOT/annealed-fragments.yaml"
```

[![The 13 fragment pairs aligned by expected base pairing](assets/annealed-fragments.svg)](assets/annealed-fragments.svg)

Each fragment occupies one row. The two oligos are aligned on the bases that
should anneal. Primer-binding, target, toehold, and barcode spans keep the same
colors across the tutorial; exposed barcode and toehold arms remain unpaired.
The strand labels give the actual order lengths.

## 4. Compare the interface strings

The planner compares the toeholds and barcodes that must remain distinguishable
within one assembly reaction. Draw the selected strings through the Python API:

```python
import os
from pathlib import Path

from dnadesign.junction import render_sequence_dissimilarity_svg

root = Path(os.environ["JUNCTION_DEMO_ROOT"])
output = root / "reviews" / "sequence-dissimilarity.svg"
output.parent.mkdir(parents=True, exist_ok=True)

# Reuse the request and write one deterministic SVG.
output.write_bytes(
    render_sequence_dissimilarity_svg(
        root / "request.yaml",
        assembly_group_id="gene-scale-example",
    )
)
```

[![Pairwise string comparisons for 12 selected interfaces](assets/sequence-dissimilarity.svg)](assets/sequence-dissimilarity.svg)

In the first two panels, a larger number means the strings differ more. In the
third, a smaller number means they share a shorter exact span. The planner uses
these sequence comparisons during search; the plots do not predict folding or
reaction yield.

## 5. Inspect each three-way interface

```bash
# Render every local junction at nucleotide resolution.
uv run baserender job run \
  "$JUNCTION_DEMO_ROOT/junction-detail.yaml"
```

[![Nucleotide-level geometry for all 12 three-way interfaces](assets/junction-detail.svg)](assets/junction-detail.svg)

Each panel shows one target helix, its perpendicular barcode helix, the
complement-strand nick, 5′/3′ direction, and every declared base pair. Use this
view to inspect local sequence geometry. Use `orders/oligos.tsv` for complete
order strings.

## 6. Follow the complete sequence path

```bash
# Render the target specification, oligos, pre-ligation state, and PCR product.
uv run baserender job run \
  "$JUNCTION_DEMO_ROOT/assembly-process.yaml"
```

[![The target, fragment oligos, pre-ligation interfaces, and expected PCR duplex](assets/assembly-process.svg)](assets/assembly-process.svg)

The first row is the submitted in-silico target specification, shown with a
dashed outline. The next rows show the physical oligos before annealing, the
expected pre-ligation geometry, and the primer-extended duplex expected from
PCR. A product stays on one row when the nucleotide scale permits it and wraps
only when it exceeds the available width.

## 7. Verify and hand off

```bash
# Reopen the finished bundle and reproduce its sequences and checks.
uv run junction verify "$JUNCTION_DEMO_ROOT/design-bundle" --format json
```

| Need | File |
| --- | --- |
| Order rows | `orders/oligos.tsv` |
| Targets, oligos, and expected PCR products | `sequences/*.fasta` |
| Selected loci, strings, and search receipts | `plan.json` |
| Passed and `not_run` checks | `checks.json` |
| File inventory, digests, and replay identity | `manifest.json` |

Next, use [Prepare a request](guides/prepare-a-request.md) for real targets or
[Inspect and verify](guides/inspect-and-verify.md) for a field-by-field bundle
review.
