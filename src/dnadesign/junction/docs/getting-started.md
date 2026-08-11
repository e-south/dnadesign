---
doc_id: junction-getting-started
title: Run the junction gene-scale example
type: tutorial
audience: first-time users
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Run the gene-scale example

This tutorial plans one 705 bp target as 13 fragment pairs joined at 12
three-way interfaces. It produces order and sequence files, verifies the
bundle, and renders four optional checks.

## 1. Choose the input

`junction` accepts a complete JSON or YAML request. The `request` command can
also place raw, text, or FASTA sequences into an existing planning profile. A
FASTA input has this form (the sequence is shortened here only for display):

```fasta
>gene-scale-target
TCATTCTAGGCTAGCTGAGT...TATAGGGTTCTGCAGTACTCGGTCCTG
```

Use the checked-in request for the runnable example:

```bash
uv sync --locked
JUNCTION_DEMO_ROOT="$(mktemp -d)"
JUNCTION_DEMO_ROOT="$(cd "$JUNCTION_DEMO_ROOT" && pwd -P)"
export JUNCTION_DEMO_ROOT
cp src/dnadesign/junction/examples/gene-scale/request.yaml \
  "$JUNCTION_DEMO_ROOT/request.yaml"
cp src/dnadesign/junction/examples/gene-scale/jobs/*.yaml \
  "$JUNCTION_DEMO_ROOT/"
```

For your own FASTA file, replace the targets in a reviewed base request:

```bash
uv run junction request \
  --base-request reviewed-request.yaml \
  --input targets.fasta \
  --assembly-group assembly-01 \
  --primer-binding-length 20 \
  > request.json
```

FASTA record IDs become target IDs. The compiler normalizes whitespace and
case, but rejects ambiguity codes, duplicate IDs, and malformed input. It
preserves the base request's planning and order policy; inspect those fields
before using the compiled request.

## 2. Build the design bundle

```bash
uv run junction build "$JUNCTION_DEMO_ROOT/request.yaml" \
  --output "$JUNCTION_DEMO_ROOT/design-bundle" \
  --format json
```

The search chooses one target-derived toehold at each interface, generates
temporary barcodes, and compares the selected strings within their assembly
group. It then composes both strands of every fragment and checks exact target
reconstruction, primer matches, and the declared order-length interval.

The output directory is create-only. Reuse requires a new destination rather
than overwriting an earlier result.

## 3. Check the fragment pairs

Run the packaged BaseRender job:

```bash
uv run baserender job run \
  "$JUNCTION_DEMO_ROOT/annealed-fragments.yaml"
```

[![The 13 fragment pairs aligned by expected base pairing](assets/annealed-fragments.svg)](assets/annealed-fragments.svg)

Each row shows the two orderable strands for one fragment. Vertical alignment
marks declared Watson–Crick pairs; the external barcode and target-derived
toehold arms remain exposed. The labels report the actual strand lengths.

## 4. Compare the selected interfaces

The search uses three string comparisons: position-weighted edit distance for
toeholds, edit distance for barcodes, and the longest shared span after each
toehold is joined to its assigned barcode. Render the selected set through the
public API:

```python
from pathlib import Path
import os

from dnadesign.junction import render_sequence_dissimilarity_svg

root = Path(os.environ["JUNCTION_DEMO_ROOT"])
request = root / "request.yaml"
output = root / "reviews" / "sequence-dissimilarity.svg"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(
    render_sequence_dissimilarity_svg(
        request,
        assembly_group_id="gene-scale-example",
    )
)
```

[![Pairwise string comparisons for 12 selected interfaces](assets/sequence-dissimilarity.svg)](assets/sequence-dissimilarity.svg)

Larger values indicate better separation in the first two panels; smaller
values indicate better separation in the last. These are the planner's string
metrics, not folding or thermodynamic predictions.

## 5. Inspect every three-way interface

```bash
uv run baserender job run \
  "$JUNCTION_DEMO_ROOT/junction-detail.yaml"
```

[![Nucleotide-level geometry for all 12 three-way interfaces](assets/junction-detail.svg)](assets/junction-detail.svg)

Each panel shows the horizontal target helix, the perpendicular barcode helix,
the complement-strand nick, strand direction, and exact paired bases. The
figure is a local sequence check; the bundle remains the source for complete
oligo strings and search evidence.

## 6. Review the complete oligo plan

```bash
uv run baserender job run \
  "$JUNCTION_DEMO_ROOT/assembly-process.yaml"
```

[![The target, fragment oligos, pre-ligation interfaces, and expected PCR duplex](assets/assembly-process.svg)](assets/assembly-process.svg)

This view aligns the input target, separate fragment oligos, complete
pre-ligation assembly, and expected PCR duplex on nucleotide coordinates. Use
the other two molecular views when exact pairing or local junction geometry is
the review question.

## 7. Verify and hand off

```bash
uv run junction verify "$JUNCTION_DEMO_ROOT/design-bundle" --format json
```

Use these files for the next task:

| Need | File |
| --- | --- |
| Complete order rows | `orders/oligos.tsv` |
| Target, oligo, and expected PCR sequences | `sequences/*.fasta` |
| Selected loci, strings, and search receipts | `plan.json` |
| Compact pass and `not_run` states | `checks.json` |
| Replay identity and file digests | `manifest.json` |

Continue with [Prepare a request](guides/prepare-a-request.md) for real inputs
or [Inspect and verify](guides/inspect-and-verify.md) for the complete bundle
review path.
