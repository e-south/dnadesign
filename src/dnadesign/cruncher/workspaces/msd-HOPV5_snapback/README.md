# MSD-HOPV5 Snapback visual workspace

This workspace is a visual-only sibling of the released-product Snapback solve lane. It does not claim to be a
current 0/3/3 dual-enzyme solution, and it does not participate in catalog search. It renders one explicit prior
MSD-HOPV5 design so it can be compared against generated solve outputs without mixing artifacts.

The checked-in spec decomposes `CCTCAGCCCGCTGA` as:

- precursor nickase site: `CCTNAGC`, with the motif-degenerate `N` resolved to `C` in the concrete precursor
- nick boundary from left: `2`
- exposed bottom product: `GG` + `AGTC` + `GGGC` + `GACT`
- stem: `AGTC`
- cap: `GGGC`
- foldback arm: `GACT`
- effective foldback stem shown in the folded view: `CC` + `AGTC` = `6 bp`

Generate the visual:

```bash
uv run cruncher snapback visual \
  --spec src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback/configs/snapback/msd-HOPV5.visual.snapback.yaml \
  --force-overwrite
```

Outputs are generated under `outputs/` and intentionally ignored.
