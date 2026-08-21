---
doc_id: construct-annotated-sequence-parts
title: Annotated sequence-part placement
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-21
---

# Annotated sequence-part placement

Use `AnnotatedSequencePartV1` when another system has already authored a
sequence and its nested features. Construct treats the supplied part as one
object. It validates the shared contract, places the sequence, offsets the
nested coordinates, and preserves source digests. It does not recompute the
part's sequence, annotations, or scientific meaning.

The contract requires:

- a sequence and matching SHA-256 digest;
- explicit `strandedness` and `topology`, including `not_asserted` when the
  producer has not modeled a physical molecule;
- zero-based, half-open feature coordinates whose sequences match the part;
- digest-pinned source records or artifacts.

Call `dnadesign.construct.place_annotated_part()` with an explicit linear
template interval and orientation. The result records both source and realized
feature coordinates. Reverse-complement placement transforms realized feature
coordinates and sequences while retaining source coordinates and digests.

This operation is not an assembly-readiness decision. Destination-specific
restriction sites, homology ends, vector identity, and biological acceptance
remain caller inputs. Circular parts require a separate linearization contract
and are rejected here.
