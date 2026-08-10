---
doc_id: junction-package
title: junction
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

![junction banner](assets/junction-banner.svg)

`junction` converts exact linear DNA targets into a reviewable oligo design for
Sidewinder-style three-way-junction assembly. A request declares the targets,
which targets are designed together, recovery primers, search limits, and order
labels. The tool selects junction sequences, proves exact string
reconstruction, and writes a replay-verifiable bundle. It does not design
primers, predict molecular behavior, run an experiment, or place an order.

## Documentation

- [Start here](docs/README.md) for the learning, use, and reference routes.
- [Build an example](docs/getting-started.md) to follow the complete software path.
- [Understand the method](docs/explanation/how-junction-works.md) before changing search settings.
- [Prepare a request](docs/guides/prepare-a-request.md) or [review a bundle](docs/guides/inspect-and-verify.md).

## Review the design

[![Base-level details for two selected three-way junctions](docs/assets/junction-detail.svg)](docs/assets/junction-detail.svg)

BaseRender can draw exact fragment annealing, the planned molecular states, or
selected three-way junctions from the verified review record. Each
[review job](examples/three-fragment-review/jobs/) answers one question and
writes one SVG per selected target. These schematics do not predict folding,
ligation, PCR, yield, or fidelity.
