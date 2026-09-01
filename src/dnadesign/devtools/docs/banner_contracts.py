"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/banner_contracts.py

Banner contracts for documentation validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from dnadesign.devtools.ci.changes import discover_repo_tools
from dnadesign.devtools.docs.badges import rendered_markdown_images
from dnadesign.devtools.docs.banners.catalog import BANNERS
from dnadesign.devtools.docs.banners.render import check_banners
from dnadesign.devtools.docs.check_contracts import (
    TOOL_README_BANNER_DIMENSION_PATTERN,
    TOOL_README_BANNER_LABEL_PATTERN,
)
from dnadesign.devtools.docs.document_metadata import (
    _markdown_body_without_frontmatter,
)


def _resolve_readme_banner_reference(*, repo_root: Path, readme_path: Path, target_rel: str) -> tuple[Path, Path]:
    resolved_repo_root = repo_root.resolve()
    readme_relative = readme_path.relative_to(repo_root)
    declared_relative = Path(os.path.normpath(str(readme_relative.parent / target_rel)))
    if declared_relative.is_absolute() or declared_relative.parts[:1] == ("..",):
        raise ValueError(target_rel)

    target_path = (resolved_repo_root / declared_relative).resolve()
    try:
        target_path.relative_to(resolved_repo_root)
    except ValueError as error:
        raise ValueError(target_rel) from error
    return declared_relative, target_path


def _top_rendered_readme_banner(text: str) -> tuple[int, str] | None:
    for image in rendered_markdown_images(text):
        if image.line_no > 25 or TOOL_README_BANNER_LABEL_PATTERN.search(image.label) is None:
            continue
        if len(image.sources) != 1:
            return image.line_no, ""
        return image.line_no, image.sources[0]
    return None


def _find_banner_catalog_inventory_issues(repo_root: Path) -> list[str]:
    banner_source = repo_root / "src" / "dnadesign" / "devtools" / "docs" / "banners"
    if not banner_source.is_dir():
        return []

    referenced_paths_by_readme: dict[str, str] = {}
    for readme_path in sorted((repo_root / "src" / "dnadesign").rglob("README.md")):
        banner = _top_rendered_readme_banner(readme_path.read_text(encoding="utf-8"))
        if banner is None:
            continue
        _line_no, link = banner
        parsed = urlparse(link)
        if parsed.scheme or link.startswith("mailto:") or not link.lower().endswith(".svg"):
            continue
        target_rel = link.split("#", 1)[0].strip()
        if not target_rel:
            continue
        try:
            declared_relative, _target_path = _resolve_readme_banner_reference(
                repo_root=repo_root,
                readme_path=readme_path,
                target_rel=target_rel,
            )
        except ValueError:
            continue
        readme_relative = readme_path.relative_to(repo_root).as_posix()
        referenced_paths_by_readme[readme_relative] = declared_relative.as_posix()

    catalog_paths = {Path(spec.path).as_posix() for spec in BANNERS}
    referenced_paths = set(referenced_paths_by_readme.values())
    issues = [
        f"{path}: tool README banner path is not declared in the banner catalog."
        for path in sorted(referenced_paths - catalog_paths)
    ]
    issues.extend(
        f"{path}: banner catalog path is not referenced by a tool README."
        for path in sorted(catalog_paths - referenced_paths)
    )

    catalog_paths_by_readme: dict[str, str] = {}
    for spec in BANNERS:
        readme_path = Path(spec.readme_path).as_posix()
        banner_path = Path(spec.path).as_posix()
        existing_path = catalog_paths_by_readme.get(readme_path)
        if existing_path is not None:
            issues.append(
                f"{readme_path}: banner catalog declares more than one README binding ({existing_path}, {banner_path})."
            )
            continue
        catalog_paths_by_readme[readme_path] = banner_path

    for readme_path, referenced_path in sorted(referenced_paths_by_readme.items()):
        expected_path = catalog_paths_by_readme.get(readme_path)
        if expected_path is None:
            issues.append(f"{readme_path}: tool README is not bound to a banner catalog entry.")
            continue
        if referenced_path != expected_path:
            issues.append(
                f"{readme_path}: banner path must match catalog entry {expected_path}; found {referenced_path}."
            )

    for readme_path in sorted(catalog_paths_by_readme.keys() - referenced_paths_by_readme.keys()):
        issues.append(f"{readme_path}: banner catalog README is missing or does not reference a local .svg banner.")
    return issues


def _find_tool_readme_banner_issues(repo_root: Path) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    issues: list[str] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        readme_path = src_root / tool_name / "README.md"
        if not readme_path.exists():
            continue

        text = _markdown_body_without_frontmatter(readme_path.read_text(encoding="utf-8"))
        top_block = "\n".join(text.splitlines()[:25])
        banner = _top_rendered_readme_banner(text)
        if banner is None:
            issues.append(f"{readme_path}: missing top banner image markdown line with '* banner' alt text.")
            continue

        _line_no, link = banner
        parsed = urlparse(link)
        if parsed.scheme or link.startswith("mailto:") or not link.lower().endswith(".svg"):
            issues.append(f"{readme_path}: banner link must target a local .svg asset.")
            continue

        target_rel = link.split("#", 1)[0].strip()
        if not target_rel:
            issues.append(f"{readme_path}: banner link must include a relative asset path.")
            continue

        try:
            _declared_relative, target_path = _resolve_readme_banner_reference(
                repo_root=repo_root,
                readme_path=readme_path,
                target_rel=target_rel,
            )
        except ValueError:
            issues.append(f"{readme_path}: banner asset target escapes the repository: {target_rel}.")
            continue
        if not target_path.exists():
            issues.append(f"{readme_path}: banner asset target does not exist: {target_rel}.")
            continue

        banner_text = target_path.read_text(encoding="utf-8")
        if TOOL_README_BANNER_DIMENSION_PATTERN.search(banner_text) is None:
            issues.append(
                f"{target_path}: tool banner must use the low-clutter 1200x180 SVG contract "
                'with viewBox="0 0 1200 180".'
            )

        if "placeholder" in top_block.lower():
            issues.append(f"{readme_path}: banner copy must not use placeholder wording.")

    issues.extend(_find_banner_catalog_inventory_issues(repo_root))
    return issues


def _find_banner_source_drift_issues(repo_root: Path) -> list[str]:
    banner_source = repo_root / "src" / "dnadesign" / "devtools" / "docs" / "banners"
    if not banner_source.is_dir():
        return []
    try:
        stale_paths = check_banners(repo_root)
    except ValueError as error:
        return [str(error)]
    return [
        f"{relative_path}: checked-in banner differs from its deterministic source; "
        "run 'uv run python -m dnadesign.devtools.docs.banners --repo-root .'."
        for relative_path in stale_paths
    ]
