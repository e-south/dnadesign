![junction banner](assets/junction-banner.svg)

`junction` plans oligos for assembling exact linear DNA targets through
three-way junctions. Its input is a complete, versioned request: target
sequences, assembly groups, caller-supplied recovery primers, search settings,
and order labels. It does not accept a bare sequence or design primers. For
each request, `junction` selects target-derived toeholds, generates and
assigns external barcodes, lays out the two strands for every fragment, checks
that the submitted target and its reverse complement can be reconstructed as
strings, and writes vendor-neutral order rows. A create-only bundle preserves
the request, plan, checks, order table, review data, and content identities. The
implementation is independently developed and inspired by the Sidewinder
method; it does not run the authors' PyWinder software, thermodynamic analysis,
laboratory assembly, PCR, cloning, or ordering.

## Documentation

- [Documentation index](docs/README.md): choose a learning, use, reference, or
  operations route.
- [Getting started](docs/getting-started.md): build and verify one small
  software demonstration.
- [How `junction` works](docs/explanation/how-junction-works.md): learn the
  physical idea, software model, and vocabulary before reading formulas.
- [Prepare a request](docs/guides/prepare-a-request.md): choose assembly groups,
  primers, settings, and order labels explicitly.
- [Inspect and verify](docs/guides/inspect-and-verify.md): find the evidence for
  each review question.

`junction` verifies deterministic string planning and bundle integrity. It
does not establish thermodynamic orthogonality, synthesis acceptance, ligation
yield, PCR performance, experimental fidelity, or freedom to operate.
