"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/security/tracked_text_privacy.py

Reject known personal operator data in tracked text and configuration files.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_PERSONAL_TOKEN_SIGNATURES = {
    "personal_macos_home": (
        16,
        "d19e28dc76e00c106dd58b758d2b18d3c50dc661cda56d483223c3fa8e9356a9",  # pragma: allowlist secret
    ),
    "personal_gmail": (
        23,
        "21f01c91fb95a2bd992512f931e6319c7105638e40b9c31351ee53874e49cca1",  # pragma: allowlist secret
    ),
    "personal_scc_login": (
        18,
        "058d35da946725668b8b66aeacf065e92e455f43e5be37eec405fd8905049801",  # pragma: allowlist secret
    ),
    "personal_scc_project_root": (
        22,
        "493009668cf9dde49a95ee42ec9071a1cd01c61fb131357be53c3736a7d700dc",  # pragma: allowlist secret
    ),
    "personal_scc_projectnb_root": (
        24,
        "6c9ee308fae22a05b1c658ae98762ea76883b00d055123ed62b759a3d45f70ac",  # pragma: allowlist secret
    ),
}
_ACTIVE_REMOTES_CONFIG = Path("src/dnadesign/usr/remotes.yaml")
_PUBLIC_DOC_ROOT = Path("docs")
_OPS_RUNBOOK_ROOT = Path("src/dnadesign/ops/runbooks")
_USR_ROOT = Path("src/dnadesign/usr")
_PUBLIC_TEXT_SUFFIXES = {".md", ".qsub", ".sh", ".toml", ".yaml", ".yml"}


@dataclass(frozen=True, order=True)
class PrivacyIssue:
    path: Path
    line: int
    token_name: str


def _tracked_paths(repo_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {detail or 'unknown error'}")
    return tuple(Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw)


def _read_tracked_text(path: Path) -> str | None:
    content = path.read_bytes()
    if b"\0" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_operator_surface(path: Path) -> bool:
    if path == _ACTIVE_REMOTES_CONFIG:
        return True
    parts = path.parts
    if path.suffix not in _PUBLIC_TEXT_SUFFIXES:
        return False
    if "tests" in parts or "outputs" in parts:
        return False
    if path.name == "journal.md" and "dev" in parts:
        return False
    if path.is_relative_to(_PUBLIC_DOC_ROOT):
        return True
    if path.is_relative_to(Path(".agents/skills")) or path.is_relative_to(_OPS_RUNBOOK_ROOT):
        return True
    if not path.is_relative_to(_USR_ROOT):
        return "docs" in parts
    relative_to_usr = path.relative_to(_USR_ROOT)
    return relative_to_usr.name in {"AGENTS.md", "README.md", "remotes.example.yaml"} or relative_to_usr.parts[:1] == (
        "docs",
    )


def _line_has_signature(line: str, *, length: int, digest: str) -> bool:
    if len(line) < length:
        return False
    return any(
        hashlib.sha256(line[start : start + length].encode("utf-8")).hexdigest() == digest
        for start in range(len(line) - length + 1)
    )


def find_privacy_issues(repo_root: Path) -> tuple[PrivacyIssue, ...]:
    root = repo_root.resolve()
    issues: list[PrivacyIssue] = []
    for relative_path in _tracked_paths(root):
        if relative_path == _ACTIVE_REMOTES_CONFIG:
            issues.append(PrivacyIssue(relative_path, 1, "tracked_active_remotes_config"))

        if not _is_operator_surface(relative_path):
            continue

        try:
            text = _read_tracked_text(root / relative_path)
        except FileNotFoundError:
            continue
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for token_name, (length, digest) in _PERSONAL_TOKEN_SIGNATURES.items():
                if _line_has_signature(line, length=length, digest=digest):
                    issues.append(PrivacyIssue(relative_path, line_number, token_name))
    return tuple(sorted(issues))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject known personal operator data in tracked UTF-8 text and config files."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    issues = find_privacy_issues(args.repo_root)
    if issues:
        print("Tracked text privacy check failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue.path}:{issue.line}: {issue.token_name}", file=sys.stderr)
        return 1

    print("Tracked text privacy check passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
