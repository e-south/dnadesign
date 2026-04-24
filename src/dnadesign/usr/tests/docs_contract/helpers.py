"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/docs_contract/helpers.py

Shared helpers for structural USR docs contract tests.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def read_text(rel_path: str) -> str:
    return (repo_root() / rel_path).read_text(encoding="utf-8")


def load_yaml(rel_path: str):
    return yaml.safe_load(read_text(rel_path))


def markdown_links(rel_path: str) -> set[str]:
    return {match.group(1) for match in _MARKDOWN_LINK_RE.finditer(read_text(rel_path))}


def assert_markdown_links_resolve(rel_path: str, *, ignore: set[str] | None = None) -> None:
    ignore = ignore or set()
    doc_path = (repo_root() / rel_path).resolve()
    for target in markdown_links(rel_path):
        if target in ignore:
            continue
        if target.startswith("#"):
            continue
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        candidate = parsed.path
        if not candidate:
            continue
        resolved = (doc_path.parent / candidate).resolve()
        assert resolved.exists(), f"{rel_path} links to missing path: {target}"


def heading_lines(rel_path: str) -> set[str]:
    return {line.strip() for line in read_text(rel_path).splitlines() if line.lstrip().startswith("#")}


def metadata(rel_path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in read_text(rel_path).splitlines():
        line = line.strip()
        if not line.startswith("**") or ":**" not in line:
            continue
        key, value = line.split(":**", 1)
        values[key.strip("* ")] = value.strip()
    return values


def normalized_text(rel_path: str) -> str:
    return " ".join(read_text(rel_path).split())
