---
doc_id: junction-package
title: junction
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

![junction banner](assets/junction-banner.svg)

`junction` designs vendor-neutral oligo sets for Sidewinder-style
three-way-junction assembly. A request supplies exact linear DNA targets,
assembly groups, recovery primers, search bounds, and order labels. The
deterministic planner chooses target-derived toeholds and external barcodes,
proves exact reconstruction, and writes a replay-verifiable review bundle.

## Documentation

- [Start here](docs/README.md) for the learning, use, and reference routes.
- [Build an example](docs/getting-started.md) to follow the complete software path.
- [Understand the method](docs/explanation/how-junction-works.md) before changing search settings.
- [Prepare a request](docs/guides/prepare-a-request.md) or [review a bundle](docs/guides/inspect-and-verify.md).

## Review the design

[![Separate oligos, modeled pre-ligation junctions, and the expected PCR duplex](docs/assets/assembly-process.svg)](docs/assets/assembly-process.svg)

BaseRender provides three opt-in views from the same verified record: fragment
annealing, the path from separate oligos to the expected recovered duplex, and
nucleotide-level details for selected junctions. Each [review
job](examples/three-fragment-review/jobs/) answers one question and writes one
SVG per target.
