<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL docs](../../index.md) / [Prediction and selection](../README.md)

# Prepare features and measurements

Feature transforms turn each candidate's stored vector into the matrix used for
fitting and prediction. Measurement transforms convert incoming observation
tables into the campaign's Y values. Optional training-time Y operations rescale
those values for fitting and reverse the transformation before scoring.

## Pages

- [Feature transforms](x.md): `transforms_x` and the required matrix shape.
- [Measurement and training transforms](y.md): `transforms_y`, input columns and reversible `training.y_ops`.

The [campaign walkthrough](../../workflows/campaign-round.md) uses identity
features and one measured scalar per candidate.
