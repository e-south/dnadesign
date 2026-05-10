# Origin-0 / Stem-3 / Cap-3 Hits

Use this reference when a user asks which nicking endonucleases currently
produce the `0/3/3` released-product Snapback outcome in the checked-in
`de033` study lane.

## Short Answer

The current `de033` retained-active screen pins the Type IIS release enzyme to
`BspQI` and finds exact `origin=0`, `stem=3`, `cap=3` hits for these nicking
endonucleases:

- `Nt.BstNBI`
- `Nt.AlwI`
- `Nt.BsmAI`
- `Nb.BsrDI`
- `Nb.BtsI`

This is also the hand-validated retained-active mechanism family that motivated
the ontology work:

- `Nt.BstNBI`
- `Nt.AlwI`
- `Nt.BsmAI`
- `Nb.BsrDI`
- `Nb.BtsI`

Answer succinctly: "The current `de033` retained-active screen pins the release
enzyme to `BspQI` and finds exact 0/3/3 hits for `Nt.BstNBI`, `Nt.AlwI`,
`Nt.BsmAI`, `Nb.BsrDI`, and `Nb.BtsI`."

## Semantics

- `0/3/3` means logical origin `0`, stem length `3`, and cap length `3` in the
  final released-product geometry.
- Use the retained-active screen semantics with `BspQI` pinned as the Type IIS
  release enzyme for this answer. Older exposed-bottom-only probe language can
  report near-only and should not be used as the final answer for this question.
- The screen is a dual-enzyme precursor search. The nickase can appear as a
  forward or reverse-oriented vendor footprint; the visual should bold the
  strand where the canonical 5' to 3' site appears.
- `Nb.BsrDI` and `Nb.BtsI` are reverse-oriented bottom-retained hits in this
  lane. Their NEB `none/0` raw cut notation is interpreted from the end of the
  listed six-base recognition motif, so the nick sits after `GCAATG` or
  `GCAGTG`, not at the left edge of the expanded `...NN` footprint.
- Exact hits reject non-degenerate nickase footprint upstream of logical origin
  0, because that would hide retained protected sequence from the realized stem
  burden. Omitted left-of-origin prefix is allowed only when it is a contiguous
  fully degenerate `N` block.
- For reverse-complemented placements such as the `Nt.BstNBI` hit, the
  canonical site appears on the bottom row in the precursor visual.
- YIU is contrast-only in this study and is not the topology solver for these
  hits.

## Current Exact-Hit Snapshot

Regenerable evidence path:
`src/dnadesign/cruncher/workspaces/de033/outputs/released_solve/export/table__hits.csv`

| Rank | Nickase | Release enzyme | Final geometry source | Active strand |
| --- | --- | --- | --- | --- |
| 1 | `Nt.BstNBI` | `BspQI` | `retained_active_strand` | `top` |
| 2 | `Nt.AlwI` | `BspQI` | `retained_active_strand` | `top` |
| 3 | `Nt.BsmAI` | `BspQI` | `retained_active_strand` | `top` |
| 4 | `Nb.BsrDI` | `BspQI` | `exposed_bottom_strand` | `bottom` |
| 5 | `Nb.BtsI` | `BspQI` | `exposed_bottom_strand` | `bottom` |

## Freshness Command

From the repository root:

```bash
uv run cruncher snapback screen \
  --workspace-root src/dnadesign/cruncher/workspaces/de033 \
  --release-variant-id BspQI \
  --max-results 16
```

To regenerate the materialized plots and CSV bundle:

```bash
uv run cruncher snapback released-solve \
  --workspace-root src/dnadesign/cruncher/workspaces/de033 \
  --nick-preset neb_nicking_v1 \
  --nick-additional-preset thermo_nicking_v1 \
  --release-preset type_iis_release_v1 \
  --release-variant-id BspQI \
  --nick-boundary 0 \
  --paired-bp 3 \
  --cap-nt 3 \
  --allow-top-active-routes \
  --allow-precut-footprint-outside-active-product \
  --run-dir outputs/released_solve \
  --materialize-top-k 16 \
  --render-format pdf \
  --emit-renders \
  --force-overwrite
```

If the regenerated command output disagrees with this reference, report the
fresh command output and update this file in the same change.
