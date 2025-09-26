"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/main.py

Executable entrypoint for YAML-driven inference.

Resolution order for config:
  1) --config <path> if provided
  2) ./config.yaml (current working directory)
  3) <this_dir>/config.yaml (sibling to this module)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import RootConfig
from .engine import run_extract_job, run_generate_job
from .errors import InferError
from ._logging import get_logger

_LOG = get_logger("dnadesign.infer.main")


def _default_config_search() -> Path | None:
    """Find default config.yaml in priority order: ./config.yaml → module sibling."""
    cwd_cfg = Path.cwd() / "config.yaml"
    if cwd_cfg.is_file():
        return cwd_cfg
    module_cfg = Path(__file__).with_name("config.yaml")
    if module_cfg.is_file():
        return module_cfg
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dnadesign.infer from a YAML config")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: ./config.yaml or sibling config.yaml)",
    )
    p.add_argument("--job", type=str, default="", help="Run a single job id (optional)")
    p.add_argument(
        "--dry-run", action="store_true", help="Validate config without running"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.config is not None:
        cfg_path = args.config.resolve()
    else:
        guessed = _default_config_search()
        if not guessed:
            _LOG.error(
                "No config file provided and none found. "
                "Pass --config or place a config.yaml in the CWD or next to dnadesign/infer/main.py."
            )
            sys.exit(1)
        cfg_path = guessed.resolve()

    if not cfg_path.is_file():
        _LOG.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    _LOG.info(f"Using config: {cfg_path}")

    try:
        root = RootConfig(**yaml.safe_load(cfg_path.read_text()))
    except InferError as e:
        _LOG.error(f"Config error: {e}")
        sys.exit(1)

    model = root.model
    jobs = root.jobs

    if args.job:
        jobs = [j for j in jobs if j.id == args.job]
        if not jobs:
            _LOG.error(f"Job id '{args.job}' not found in config")
            sys.exit(1)

    if args.dry_run:
        _LOG.info("✔ Config validated (dry run).")
        return

    # CLI: now supports pt_file (legacy) AND usr (new). 'records'/'sequences' remain Python-API only.
    for job in jobs:
        if job.ingest.source not in {"pt_file", "usr"}:
            _LOG.error(
                "CLI supports ingest.source in {'usr','pt_file'}. "
                "Use the Python API for 'sequences' or 'records'."
            )
            sys.exit(1)

        # For pt_file we resolve path from config dir; for usr we don't need a path.
        input_arg = (
            (cfg_path.parent / f"{job.id}.pt").as_posix()
            if job.ingest.source == "pt_file"
            else None
        )
        shown = input_arg if input_arg else f"usr:{job.ingest.dataset}"
        _LOG.info(f"▶ Running job '{job.id}' on {shown}")
        try:
            if job.operation == "extract":
                res = run_extract_job(input_arg, model=model, job=job)
            else:
                res = run_generate_job(input_arg, model=model, job=job)
        except InferError as e:
            _LOG.error(str(e))
            sys.exit(1)
        _LOG.info(f"✔ Job '{job.id}' complete. Outputs: {list(res.keys())}")


if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    print("Please run as a module, e.g. `python -m dnadesign.infer --config ...`")
    raise SystemExit(1)
