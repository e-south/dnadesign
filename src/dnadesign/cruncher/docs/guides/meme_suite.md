## MEME Suite dependency

`cruncher discover` uses the MEME Suite CLI tools (`streme`, `meme`). These are system-level dependencies (not Python), so they are installed outside of `uv`. The recommended UX is to run **cruncher** via **pixi**, which puts MEME on `PATH` while `uv` remains the Python source of truth.

### Install with pixi (recommended)

This repo ships a minimal `pixi.toml` plus a `cruncher` task that wraps `uv run cruncher`.

```bash
pixi install
pixi run cruncher -- doctor
```

Note: the pixi `meme` package provides both `meme` and `streme`. When using pixi tasks, place `-c/--config` after the subcommand (pixi inserts `--`).

### Choose MEME vs STREME

- **MEME**: best when each sequence is one site; use `--meme-mod oops` (or `zoops` if noisy).
- **STREME**: good default for larger sets or when you want speed.
- If `minw/maxw` are unset, **cruncher** derives bounds from per‑TF site lengths.

```bash
cruncher discover motifs --tool meme --meme-mod oops <config>
cruncher discover motifs --tool streme <config>
```

### Compare outputs (optional)

Run both tools into separate sources so `lock` can disambiguate, then render both logos:

```bash
cruncher discover motifs --tool streme --source-id meme_suite_streme <config>
cruncher discover motifs --tool meme --meme-mod oops --source-id meme_suite_meme <config>
cruncher catalog logos --source meme_suite_streme <config>
cruncher catalog logos --source meme_suite_meme <config>
```

After choosing, set `motif_store.source_preference` and re‑run `cruncher lock`.
Note: MEME/STREME matrices are rounded in their text output. **cruncher** renormalizes near‑1.0 rows to avoid false validation errors.

### If you don't use pixi

Point **cruncher** at a MEME Suite bin directory:

```bash
export DNADESIGN_ROOT="$(git rev-parse --show-toplevel)"
export MEME_BIN="$DNADESIGN_ROOT/.pixi/envs/default/bin"
# or set in config:
# motif_discovery:
#   tool_path: /abs/path/to/meme/bin
```

### Alternative: official MEME Suite installer

Install MEME Suite with the official installer, ensure `meme` and `streme` are on PATH,
then verify:

```bash
cruncher doctor -c <config>
```

If you're not in an activated virtualenv, prefix with `uv run`.

---

@e-south
