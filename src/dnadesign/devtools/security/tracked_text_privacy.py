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
import re
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
_PRIVATE_STUDY_TOKEN_SIGNATURES = {
    "24946af550a781fce523482d758dfe0dc4abb29ee87a5d615cc3ca2f2f065e3f",  # pragma: allowlist secret
    "406267c5f1f8d5e88e1f66610d428736253f5be43b233263d189d0ee841a3350",  # pragma: allowlist secret
    "dbe33577a145d35dbff36fdaa287352240a749886cf6deccdb68cae90c34be17",  # pragma: allowlist secret
    "cf2d78fc2c55e41542393b34031136028e82f69080ee14384ed92054e06e4a3a",  # pragma: allowlist secret
    "c494bdd3c55f7e7c3282eb86060c1899256b7118ecc8faca4783bfa8d0c33f72",  # pragma: allowlist secret
}
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{9,}")
_ACTIVE_REMOTES_CONFIG = Path("src/dnadesign/usr/remotes.yaml")
_HOME_PATH_PATTERNS = (
    re.compile("/Users" + r"/([^/\s\"']+)"),
    re.compile("/home" + r"/([^/\s\"']+)"),
    re.compile(r"[A-Za-z]:\\Users\\([^\\/\s\"']+)"),
)
_GENERIC_HOME_NAMES = {"example", "sample", "user", "username", "you"}
_SCANNED_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".qsub",
    ".sh",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_SCANNED_TEXT_NAMES = {".gitignore", ".pre-commit-config.yaml", "Dockerfile"}
_PRIVACY_MARKERS = ("/Users", "/home", "\\Users\\", "@gmail.com", "@scc", "/project")


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


def _line_has_signature(line: str, *, length: int, digest: str) -> bool:
    if len(line) < length:
        return False
    return any(
        hashlib.sha256(line[start : start + length].encode("utf-8")).hexdigest() == digest
        for start in range(len(line) - length + 1)
    )


def _line_has_nongeneric_home(line: str) -> bool:
    for pattern in _HOME_PATH_PATTERNS:
        for match in pattern.finditer(line):
            name = match.group(1).strip().lower()
            if name in _GENERIC_HOME_NAMES or name.startswith(("$", "<", "{")):
                continue
            return True
    return False


def _has_private_study_identifier(value: str) -> bool:
    return any(
        hashlib.sha256(candidate.encode("utf-8")).hexdigest() in _PRIVATE_STUDY_TOKEN_SIGNATURES
        for candidate in _IDENTIFIER_PATTERN.findall(value)
    )


def find_privacy_issues(repo_root: Path) -> tuple[PrivacyIssue, ...]:
    root = repo_root.resolve()
    issues: list[PrivacyIssue] = []
    for relative_path in _tracked_paths(root):
        if _has_private_study_identifier(relative_path.as_posix()):
            issues.append(PrivacyIssue(relative_path, 1, "private_study_identifier"))
        if relative_path == _ACTIVE_REMOTES_CONFIG:
            issues.append(PrivacyIssue(relative_path, 1, "tracked_active_remotes_config"))

        if relative_path.suffix.lower() not in _SCANNED_TEXT_SUFFIXES and relative_path.name not in _SCANNED_TEXT_NAMES:
            continue

        try:
            text = _read_tracked_text(root / relative_path)
        except FileNotFoundError:
            continue
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _has_private_study_identifier(line):
                issues.append(PrivacyIssue(relative_path, line_number, "private_study_identifier"))
            if not any(marker in line for marker in _PRIVACY_MARKERS):
                continue
            has_known_signature = False
            for token_name, (length, digest) in _PERSONAL_TOKEN_SIGNATURES.items():
                if _line_has_signature(line, length=length, digest=digest):
                    issues.append(PrivacyIssue(relative_path, line_number, token_name))
                    has_known_signature = True
            if not has_known_signature and _line_has_nongeneric_home(line):
                issues.append(PrivacyIssue(relative_path, line_number, "absolute_home_path"))
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
