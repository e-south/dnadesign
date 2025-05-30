## cruncher

**cruncher** is a pipeline for multi-TF motif design, optimizer-driven sequence search, and result auditing.

1. **Parse** – read one or many TF PWMs / PSSMs (MEME, JASPAR, …) and make logo plots.
2. **Sample** – build overlapping-motif sequences and run a discrete Gibbs (or other) optimiser to find high-scoring hits.
3. **Analyze** – reload any past batch and generate extra plots/tables without re-sampling.


> **Quick start**
>
> ```bash
> # 1. sanity-check motifs
> cruncher parse   configs/example.yaml
>
> # 2. run optimiser
> cruncher sample  configs/example.yaml
>
> # 3. open   results/batch_<timestamp>/plots/score_kde.png
> ```

---

#### Project layout (core folders)

```
dnadesign/
├─ configs/            # your YAMLs (examples inside)
└─ cruncher/
   ├─ motif/           # PWM dataclass + plug-in parsers
   ├─ sample/          # state, scorer, optimisers (Gibbs, SA, …)
   ├─ plots/           # PSSM logos + MCMC diagnostics
   ├─ persistence/     # save/load csv, trace.nc, etc.
   └─ results/         # auto-created batch folders
```

#### Minimal config

```yaml
cruncher:
  mode: sample                    # parse | sample | analyse
  out_dir: results/               # where all outputs land
  regulator_sets:                 # list of regulator sets to use   
    - [cpxR, soxR]                # set #1: two TFs
    # - [crp, fis, ihf]           # set #2: three TFs
  motif:                          # parse-specific config
    formats:
      .txt: MEME
      .pfm:  JASPAR
    plot:
      logo:      true
      bits_mode: information
      dpi:       200
  sample:
    bidirectional: true           # score sequences based on both forward and reverse-complement (take max per-PWM)
    init:
      kind: random                # random, consensus_shortest, consensus_longest, or integer length 
      pad_with: background_pwm    # background (uniform iid), background_pwm (sample i.i.d. from overall PWM base frequencies)
    optimiser:
      kind: gibbs
      gibbs:
        draws:    400             # recorded Gibbs sweeps; each produces one full-sequence sample
        tune:     0               # “burn-in” Gibbs sweeps not recorded—allows chain to move toward high-probability region
        beta:     0.01            # higher β sharpens acceptance toward higher‐score sequences
        block_size: 5             # how many adjacent sites to update in one Gibbs move
        swap_prob:  0.5           # probability of doing an MH substring-swap instead of block-Gibbs
        chains:   4               # independent Gibbs chains to assess convergence
        cores:    4
        min_dist: 1               # minimum Hamming distance between final reported elites (diversity constraint)
    top_k: 200                    # number of unique, diverse sequences to return
    plots:
      trace:       true
      autocorr:    true
      convergence: true
      scatter_pwm: true
  analysis:                       # analysis-specific config
    runs: []                      # default = most recent sample
    plots:
      - score_kde
      - scatter_pwm
      - logo_elites
```

---

#### Example usage

| Goal   | Command  | What happens  |
| ------ | -------- | ------------- |
| **Parse (dry-run)** | `cruncher parse   configs/example.yaml`   | ✓ Parses every PWM<br>✓ Saves `{tf}_logo.png` in `results/motif_logos/`<br>✓ Prints length + information bits   |
| **Sample**  | `cruncher sample  configs/example.yaml --seed 42` | ✓ Runs optimiser for each TF pair / orientation<br>✓ Creates `results/batch_YYYYMMDDTHHMM/` containing:<br>  – `hits.csv` (top-K sequences)<br>  – `trace.nc` (ArviZ)<br>  – auto diagnostics in `plots/`<br>  – frozen `config_used.yaml` |
| **Analyse** | `cruncher analyse configs/example.yaml --run 20240525T1130` | ✓ Loads existing batch (no sampling)<br>✓ Produces extra plots chosen in `analysis.plots`   |

*(`--run` omitted ⇒ most-recent batch.)*

#### Result folder

```
results/
└─ batch_20240525T1130/
   ├─ config_used.yaml        # frozen copy of YAML
   ├─ motifs/                 # cached parsed PWM json
   ├─ hits.csv                # ranked sequences + per-TF scores
   ├─ trace.nc                # MCMC trace (ArviZ)
   ├─ plots/
   │   ├─ trace_score.png
   │   ├─ score_kde.png
   │   └─ …
   └─ README.txt              # git hash, runtime stats
```

Re-running `sample` creates a fresh timestamped batch.

---

#### Extending Cruncher

| Task   | How-to   |
| ------ | ------ |
| **Add a new PWM file** | Drop `<tf>.meme` (or `.pfm`, `.txt`) into `motifs/`, run `cruncher parse` to preview.  |
| **Support a new format**        | Create `cruncher/motif/parsers/myfmt.py`, decorate a `parse_myfmt()` function with<br>`@register("MYFMT")`.  Done. |
| **Try another optimiser**   | Set `sample.optimiser.kind` to `"sa"` (simulated annealing) or add a new subclass in `overlap/optimizer/`.    |
| **Re-plot without re-sampling** | `cruncher analyse …` pointing to any `batch_*/` folder.  |


 -e-south