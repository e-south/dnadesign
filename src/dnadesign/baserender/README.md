## baserender

Minimal rendering of biological sequences with annotations.

**Layout**:
``` bash
src/dnadesign/baserender/
├─ src/       # package
├─ jobs/      # YAML presets (call by name)
└─ results/   # outputs per job
````

### Install
`pyproject.toml`:
```toml
[project.scripts]
baserender = "dnadesign.baserender.src.cli:app"
````

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

#### Run a preset job (uses `jobs/` and writes into `results/`)

```bash
baserender job CpxR_LexA
```

#### Render a single still from a job

```bash
baserender job CpxR_LexA --rec-id <record-id> --fmt pdf
# or
baserender job CpxR_LexA --row 0 --fmt png
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

### Preset example

`src/dnadesign/baserender/jobs/cpxR_LexA.yml`

```yaml
input:
  path: /Users/.../records.parquet
  format: parquet
  columns:
    id: id
    sequence: sequence
    annotations: densegen__used_tfbs_detail
  alphabet: DNA
  # Limit how many sequences to process (default 500). Set 0 to process all.
  limit: 500

plugins:
  - sigma70

style: {}   # optional overrides (dpi, fonts, palette, etc.)

output:
  video:
    # If omitted, defaults to results/<job>/<job>.mp4
    path: /Users/.../results/CpxR_LexA/cpxR_LexA.mp4
    fmt: mp4
    fps: 2
    seconds_per_seq: 1.0
    # Pause longer on selected records (additive seconds)
    pauses: {}
    # Frame size (px); if omitted, uses first frame (size auto)
    width_px: 1400
    height_px: null
    # Optionally constrain total video length (seconds)
    # total_duration: 120
```

---

@e-south