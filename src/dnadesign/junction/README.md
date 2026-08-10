![junction banner](assets/junction-banner.svg)

`junction` designs oligos for building one or more exact DNA sequences with
Sidewinder-style three-way junctions. It chooses target-derived toeholds and
temporary barcodes, checks that the planned strands reconstruct each target
and meet declared length limits, then writes a replay-verifiable bundle with
order tables and FASTA files.

## Documentation

- [Start here](docs/README.md) to choose a task from the full documentation map.
- [Run the example](docs/getting-started.md) for a complete request, build, and
  verification.
- [Prepare a request](docs/guides/prepare-a-request.md) for one target or a
  jointly designed set.
- [Inspect a bundle](docs/guides/inspect-and-verify.md) to trace exact sequences,
  checks, and molecular views.
- [Understand the method](docs/explanation/how-junction-works.md) before changing
  geometry or search settings.

[![Input target, fragment oligos, pre-ligation junctions, and the expected PCR duplex](docs/assets/assembly-process.svg)](docs/assets/assembly-process.svg)

The process view follows the submitted target through oligo design,
pre-ligation junctions, and the expected PCR duplex. Focused views show
fragment annealing and individual junctions at nucleotide resolution.
