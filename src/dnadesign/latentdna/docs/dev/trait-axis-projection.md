# Trait Axis Projection

**Owner:** dnadesign-maintainers
**Status:** implemented
**Last verified:** 2026-08-09

`trait_axis_projection` fits a signed direction between declared low and high
cohorts in an existing representation, then projects chosen rows onto that
direction. LatentDNA owns the geometry and provenance. External studies own
cohort meaning, labels, thresholds, and downstream decisions.

## Contract

Inputs are explicit:

- one materialized representation view;
- low and high fit-cohort selectors;
- one or more score-population selectors;
- a normalization and weighting policy;
- optional endpoint-sensitivity definitions.

The row builder emits projection scores and fit provenance. The summary builder
emits counts, endpoint separation, source-value correlation when declared, and
sensitivity results. Neither builder infers scientific meaning from column
names.

For low centroid \(c_l\), high centroid \(c_h\), and row vector \(x\):

\[
a = \frac{c_h-c_l}{\lVert c_h-c_l \rVert_2}, \qquad
s(x) = (x-c_l)^\top a
\]

The fitted direction is invalid when either cohort is empty, dimensions do not
match, or the centroid difference has zero norm.

## Ownership Boundary

LatentDNA owns:

- selector evaluation and aligned-row validation;
- centroid fitting and projection math;
- deterministic row and summary artifacts;
- fit, score, and sensitivity provenance;
- generic plots and notebook controls.

The caller owns:

- what the endpoints mean;
- which rows may fit or only be scored;
- whether observations are independent;
- acceptance thresholds and ranking policy;
- interpretation and publication claims.

Study configuration stays in the external study workspace. Public integration
rules are documented in [study workspaces](../../../../../docs/integrations/study-workspaces.md).

## Runtime Surfaces

- Builder: `src/dnadesign/latentdna/src/scalars/builders/trait_axis_projection.py`
- Dispatch: `src/dnadesign/latentdna/src/scalars/build.py`
- Contract tests: `src/dnadesign/latentdna/tests/test_trait_axis_projection.py`

The primitive uses the normal scalar build path. It does not add a study CLI,
alternate persistence layer, or hidden workspace discovery rule.

## Fail-Closed Rules

Reject the build before publication when:

- a required view, selector, key, or numeric field is missing;
- row ledgers and matrices do not align;
- fit cohorts overlap when the policy forbids overlap;
- a fit cohort has no finite rows;
- a fitted axis is degenerate;
- an artifact path escapes the workspace output root;
- requested sensitivity results cannot be tied to their endpoint definition.

Do not silently pool values from separately calibrated sources. Fit separate
axes and compare their directions only when view, dimensionality, and
normalization match.

## Performance

Load each matrix once per build, use boolean masks for cohort selection, and
project score populations in batches when their size would exceed the
workspace memory budget. Do not retain duplicate full matrices for each axis.
The existing workspace memory policy remains authoritative.

## Validation

```bash
uv run pytest -q src/dnadesign/latentdna/tests/test_trait_axis_projection.py
uv run latentdna validate workspace --workspace /path/to/workspace --deep --json
uv run latentdna workspace snapshot --workspace /path/to/workspace --json --dry-run
```

Generated tables, plots, notebooks, and deliverables must be produced through
declared workspace recipes or official CLI commands. Do not hand-edit outputs.
