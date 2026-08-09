---
doc_id: junction-package
title: junction
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

![junction banner](assets/junction-banner.svg)

`junction` turns exact linear DNA targets into oligo plans for Sidewinder-style
three-way-junction assembly. You provide the targets, grouping, recovery
primers, search limits, and order labels. The tool selects sequence junctions,
checks exact reconstruction, and writes a reproducible bundle. It does not
predict molecular behavior, design primers, run PCR, or place orders.

## Documentation

- [Documentation index](docs/README.md): choose a learning, use, or reference
  route.
- [Getting started](docs/getting-started.md): build and verify one gene-scale
  software example.
- [How `junction` works](docs/explanation/how-junction-works.md): learn the
  physical idea, software model, and vocabulary before reading formulas.
- [Prepare a request](docs/guides/prepare-a-request.md): choose assembly groups,
  primers, settings, and order labels explicitly.
- [Inspect and verify](docs/guides/inspect-and-verify.md): find the evidence for
  each review question.

## Nucleotide review

[![Nucleotide-level BaseRender audit of a synthetic three-fragment junction design](docs/assets/three-fragment-review.svg)](docs/assets/three-fragment-review.svg)

Open the map at full size to follow the exact order strands, annealed fragment
pairs, `t/t*` and `b/b*` interfaces, recovered duplex, and recovery primers.
Light gray lines mark declared Watson–Crick pairs. The image comes from the
checked-in 170 bp, three-fragment request through the same typed review contract
used for larger plans. It is a sequence audit, not a molecular simulation.
