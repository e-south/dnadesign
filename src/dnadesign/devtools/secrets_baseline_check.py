"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/secrets_baseline_check.py

Validates detect-secrets baseline entries so stale file-path allowlists fail CI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _path_is_within(*, parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_baseline(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"detect-secrets baseline file not found: {path}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in detect-secrets baseline {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Detect-secrets baseline payload must be an object: {path}")
    return payload


def _iter_baseline_result_paths(payload: dict) -> list[str]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("Detect-secrets baseline must include a 'results' object.")

    paths: list[str] = []
    for path_value in sorted(results):
        normalized = str(path_value or "").strip()
        if not normalized:
            raise ValueError("Detect-secrets baseline contains an empty result path.")
        paths.append(normalized)
    return paths


def _resolve_result_path(*, repo_root: Path, result_path: str) -> Path:
    candidate = Path(result_path.replace("\\", "/"))
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()
    if not _path_is_within(parent=repo_root, child=resolved):
        raise ValueError(f"Detect-secrets baseline path escapes repository root: {result_path}")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate detect-secrets baseline file-path hygiene.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--baseline", type=Path, default=Path(".secrets.baseline"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    baseline_path = args.baseline if args.baseline.is_absolute() else (repo_root / args.baseline).resolve()

    try:
        baseline_payload = _load_baseline(baseline_path)
        result_paths = _iter_baseline_result_paths(baseline_payload)
    except ValueError as exc:
        print(str(exc))
        return 1

    missing: list[str] = []
    for result_path in result_paths:
        try:
            resolved = _resolve_result_path(repo_root=repo_root, result_path=result_path)
        except ValueError as exc:
            print(str(exc))
            return 1
        if not resolved.exists():
            missing.append(result_path)

    if missing:
        print("Detect-secrets baseline includes paths that no longer exist in the repository:")
        for path_value in missing:
            print(f"- {path_value}")
        return 1

    print(f"Detect-secrets baseline path hygiene passed ({len(result_paths)} tracked paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
