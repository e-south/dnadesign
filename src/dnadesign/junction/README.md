---
doc_id: junction-package
title: junction
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

![junction banner](assets/junction-banner.svg)

`junction` makes an oligo plan for assembling exact linear DNA targets with
Sidewinder-style three-way junctions. You supply the targets, target groups,
recovery primers, search settings, and order labels. `junction` chooses the
junction sequences, checks that the plan reconstructs each target, and writes a
verified bundle. It does not design primers, predict molecular behavior, run an
experiment, or place an order.

## Documentation

- [Start here](docs/README.md) for the learning, use, and reference routes.
- [Build an example](docs/getting-started.md) to see the complete software path.
- [Understand the method](docs/explanation/how-junction-works.md) before changing search settings.
- [Prepare a request](docs/guides/prepare-a-request.md) or [review a bundle](docs/guides/inspect-and-verify.md).

## Review the design

[![Base-level details for two selected three-way junctions](docs/assets/junction-detail.svg)](docs/assets/junction-detail.svg)

BaseRender can draw selected fragment pairs, a target-scale assembly map, or
selected three-way junctions from the verified review record. The
[review jobs](examples/three-fragment-review/jobs/) generate only what you ask
for. These schematics do not predict folding, ligation, PCR, yield, or fidelity.
