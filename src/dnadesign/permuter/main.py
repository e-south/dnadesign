"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/main.py

Entry-point CLI.

Usage:
  - Single config (back-compat):
      python -m dnadesign.permuter.main --config dnadesign/permuter/experiments/exp_a/config.yaml

  - Workspace mode:
      python -m dnadesign.permuter.main --config dnadesign/permuter/workspace.yaml
      python -m dnadesign.permuter.main --config dnadesign/permuter/workspace.yaml --only exp_a,exp_b
      python -m dnadesign.permuter.main --config dnadesign/permuter/workspace.yaml --list

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dnadesign.permuter.logging_utils import init_logger
from dnadesign.permuter.runner import run_config
from dnadesign.permuter.workspace import (
    is_workspace_config,
    iter_enabled,
    load_workspace,
)

_LOG = init_logger("INFO")


def _autodetect_config() -> Path | None:
    """
    Search order:
      1) ./workspace.yaml
      2) ./config.yaml
      3) <module_dir>/workspace.yaml
      4) <module_dir>/config.yaml
    """
    here = Path.cwd()
    moddir = Path(__file__).parent
    for cand in [
        here / "workspace.yaml",
        here / "config.yaml",
        moddir / "workspace.yaml",
        moddir / "config.yaml",
    ]:
        if cand.is_file():
            return cand.resolve()
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the permuter pipeline")
    p.add_argument(
        "--config",
        type=Path,
        default=None,  # <-- allow autodetect
        help="YAML file: single experiment config OR workspace.yaml (autodetect if omitted)",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated experiment names (workspace mode)",
    )
    p.add_argument("--list", action="store_true", help="List experiments and exit")
    p.add_argument(
        "--base-output",
        type=Path,
        default=None,
        help="Override output root (optional; workspace mode defaults to experiment dir)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure in workspace mode",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = args.config.resolve() if args.config is not None else _autodetect_config()

    if cfg is None or not cfg.is_file():
        _LOG.error(
            "No config found. Pass --config or place one of:\n"
            "  - ./workspace.yaml\n"
            "  - ./config.yaml\n"
            f"  - {Path(__file__).parent / 'workspace.yaml'}\n"
            f"  - {Path(__file__).parent / 'config.yaml'}"
        )
        sys.exit(1)

    # Workspace mode
    if is_workspace_config(cfg):
        exp_root, exps = load_workspace(cfg)
        if args.list:
            for e in exps:
                mark = "[x]" if e.enabled else "[ ]"
                rel = e.config_path.relative_to(exp_root)
                print(f"{mark} {e.name:20s}  ({rel})")
            return

        only = [s for s in args.only.split(",") if s.strip()] if args.only else None
        errors = 0
        for exp in iter_enabled(exps, only):
            try:
                run_config(
                    exp.config_path,
                    base_output=args.base_output or exp.dir,
                )
            except Exception as exc:  # pragma: no cover
                errors += 1
                _LOG.error(f"[{exp.name}] FAILED: {exc}")
                if args.fail_fast:
                    break
        if errors:
            sys.exit(1)
        _LOG.info("✔ All selected experiments completed successfully.")
        return

    # Single-config mode (back-compat)
    try:
        run_config(cfg, base_output=args.base_output or cfg.parent)
    except Exception as exc:  # pragma: no cover
        _LOG.error(str(exc))
        sys.exit(1)
    _LOG.info("✔ Done")


if __name__ == "__main__":
    main()
