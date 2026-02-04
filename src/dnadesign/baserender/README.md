## baserender

Minimal rendering of biological sequences with annotations.

**Layout**:
``` bash
src/dnadesign/baserender/
├─ src/       # package
├─ jobs/      # YAML presets (call by name)
├─ styles/    # YAML style presets
└─ results/   # outputs per job
```

### Install
`pyproject.toml`:

```toml
[project.scripts]
baserender = "dnadesign.baserender.src.cli:app"
```

### System dependencies

* **FFmpeg** (required for video export). Confirm:

  ```bash
  which ffmpeg
  ffmpeg -version
  ```

  - On macOS (Homebrew): `brew install ffmpeg`
  - On Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
  - On Conda (any OS): `conda install -c conda-forge ffmpeg`

**Compatibility:** Videos are encoded as **H.264** with **yuv420p** pixel format and `+faststart`, producing `.mp4` files that open in QuickTime Player and are drag‑and‑drop compatible with PowerPoint.

### CLI quick start

#### Help

```bash
baserender --help
```

#### Run a Job (uses `jobs/` and writes into `results/`)

```bash
baserender job run jobs/foo.yml
# or by name (looked up in jobs/):
baserender job run foo
```

#### Direct (dataset → images), with progress and plugin(s)

```bash
baserender render /path/to/records.parquet \
  --out-dir ./out/images \
  --plugin sigma70 \
  --limit 500   # default; set 0 for all
```

### Dataset contract

* `sequence` (str)
* `densegen__used_tfbs_detail` (list of dicts: `{"offset": int, "orientation": "fwd"|"rev", "tf": str, "tfbs": str}`)
* optional `id` (str)

### Selection (CSV-driven)

Jobs can optionally select specific records via a CSV:

```yaml
selection:
  path: selections.csv
  match_on: id              # id | sequence | row
  column: id                # CSV column to read
  overlay_column: details   # optional text overlay column (must exist if set)
  keep_order: true
  on_missing: warn          # skip | warn | error
```

Row selection uses **Parquet row index (0‑based)** from the dataset.

### Style presets

Style presets live in `styles/`. The default is `presentation_default.yml`.
Use `baserender style list` to see available presets and `baserender style show` to inspect the effective mapping.

### Output path resolution

Output paths are resolved relative to `results_dir/<job_name>/`:

* Absolute paths are used as‑is.
* Relative paths resolve under `results_dir/<job_name>/...`.

### Preset example

`src/dnadesign/baserender/jobs/cpxR_LexA.yml`

```yaml
version: 2
input:
  # Relative paths resolve from the repo root when the job is under jobs/.
  path: inputs/records.parquet
  format: parquet
  columns:
    id: id
    sequence: sequence
    annotations: densegen__used_tfbs_detail
  alphabet: DNA
  # Limit how many sequences to process (default 500). Set 0 to process all.
  limit: 500

pipeline:
  plugins:
    - sigma70

style:
  preset: presentation_default
  overrides: {}

output:
  video:
    # If omitted, defaults to results/<job>/<job>.mp4
    path: results/CpxR_LexA/cpxR_LexA.mp4
    fmt: mp4
    fps: 2
    frames_per_record: 1
    pauses: {}
    width_px: 1400
    height_px: null
  images:
    dir: results/CpxR_LexA/images
    fmt: png
```

---

@e-south
