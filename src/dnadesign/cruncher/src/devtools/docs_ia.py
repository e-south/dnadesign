"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/devtools/docs_ia.py

Generate and sync Cruncher docs map and runbook-step reference sections from a
catalog and workspace machine runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

DOCS_MAP_MARKER = "docs:map"
RUNBOOK_STEPS_MARKER = "docs:runbook-steps"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def docs_root() -> Path:
    return _package_root() / "docs"


def workspaces_root() -> Path:
    return _package_root() / "workspaces"


def catalog_path() -> Path:
    return docs_root() / "meta" / "docs_catalog.yaml"


@dataclass(frozen=True)
class CatalogPage:
    path: str
    title: str
    kind: str
    audience: tuple[str, ...]
    applies_to: str
    last_verified: str
    next_paths: tuple[str, ...]


@dataclass(frozen=True)
class DocsCatalog:
    pages: tuple[CatalogPage, ...]


def load_docs_catalog(path: Path | None = None) -> DocsCatalog:
    cfg_path = catalog_path() if path is None else path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Docs catalog not found: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("docs catalog must be a mapping")
    root = payload.get("docs_catalog")
    if not isinstance(root, dict):
        raise ValueError("docs catalog missing root key: docs_catalog")
    if int(root.get("version", 0)) != 1:
        raise ValueError("docs catalog version must be 1")
    raw_pages = root.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("docs catalog pages must be a non-empty list")
    pages: list[CatalogPage] = []
    for item in raw_pages:
        if not isinstance(item, dict):
            raise ValueError("docs catalog page entries must be mappings")
        next_paths = item.get("next", [])
        if next_paths is None:
            next_paths = []
        if not isinstance(next_paths, list):
            raise ValueError("docs catalog page.next must be a list")
        audience = item.get("audience", [])
        if not isinstance(audience, list) or not audience:
            raise ValueError("docs catalog page.audience must be a non-empty list")
        pages.append(
            CatalogPage(
                path=str(item.get("path", "")).strip(),
                title=str(item.get("title", "")).strip(),
                kind=str(item.get("kind", "")).strip(),
                audience=tuple(str(value).strip() for value in audience if str(value).strip()),
                applies_to=str(item.get("applies_to", "")).strip(),
                last_verified=str(item.get("last_verified", "")).strip(),
                next_paths=tuple(str(value).strip() for value in next_paths if str(value).strip()),
            )
        )
    return DocsCatalog(pages=tuple(pages))


def _replace_marker_block(text: str, marker: str, content: str) -> str:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    if start not in text or end not in text:
        raise ValueError(f"missing required marker block: {marker}")
    left, right = text.split(start, maxsplit=1)
    middle, tail = right.split(end, maxsplit=1)
    _ = middle
    return f"{left}{start}\n{content}\n{end}{tail}"


def _write_if_changed(path: Path, new_text: str) -> None:
    old_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if old_text != new_text:
        path.write_text(new_text, encoding="utf-8")


def _render_docs_map(catalog: DocsCatalog) -> str:
    page_by_path = {page.path: page for page in catalog.pages}
    flow_sections: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "Run End-to-End Workflows",
            ("demos/demo_pairwise.md", "demos/demo_multitf.md", "demos/project_all_tfs.md"),
        ),
        (
            "Ingest and Prepare Inputs",
            ("guides/ingestion.md", "guides/meme_suite.md", "guides/troubleshooting.md"),
        ),
        (
            "Optimize and Analyze Outputs",
            (
                "guides/intent_and_lifecycle.md",
                "guides/sampling_and_analysis.md",
                "reference/artifacts.md",
            ),
        ),
        (
            "Run Studies and Portfolio Aggregation",
            (
                "guides/studies.md",
                "guides/study_length_vs_score.md",
                "guides/study_diversity_vs_score.md",
                "guides/portfolio_aggregation.md",
            ),
        ),
        (
            "Reference Contracts",
            (
                "reference/config.md",
                "reference/cli.md",
                "reference/architecture.md",
                "reference/glossary.md",
                "reference/runbook_steps.md",
                "reference/doc_conventions.md",
            ),
        ),
        (
            "Maintainer Internals",
            (
                "internals/spec.md",
                "internals/optimizer_improvements_plan.md",
                "dev/journal.md",
                "meta/style_guide.md",
            ),
        ),
    )
    lines: list[str] = []
    for heading, paths in flow_sections:
        section_lines = [f"#### {heading}"]
        for path in paths:
            page = page_by_path.get(path)
            if page is None:
                continue
            section_lines.append(f"- [{page.title}]({path})")
        if len(section_lines) > 1:
            lines.extend(section_lines)
            lines.append("")
    return "\n".join(lines).strip()


def _runbook_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for workspace in sorted(workspaces_root().iterdir(), key=lambda p: p.name):
        runbook_path = workspace / "configs" / "runbook.yaml"
        if not runbook_path.exists():
            continue
        payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid runbook payload: {runbook_path}")
        runbook = payload.get("runbook")
        if not isinstance(runbook, dict):
            raise ValueError(f"Runbook missing root key 'runbook': {runbook_path}")
        steps = runbook.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"Runbook steps must be a list: {runbook_path}")
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"Runbook step must be a mapping: {runbook_path}")
            step_id = str(step.get("id", "")).strip()
            description = str(step.get("description", "")).strip()
            run_args = step.get("run")
            if not isinstance(run_args, list) or not run_args:
                raise ValueError(f"Runbook step run args must be a non-empty list: {runbook_path}")
            command = "cruncher " + " ".join(str(token) for token in run_args)
            rows.append((workspace.name, step_id, description, command))
    return rows


def _render_runbook_steps() -> str:
    rows = [
        "| Workspace | Step ID | Description | Command |",
        "| --- | --- | --- | --- |",
    ]
    for workspace, step_id, description, command in _runbook_rows():
        rows.append(f"| `{workspace}` | `{step_id}` | {description} | `{command}` |")
    return "\n".join(rows)


def sync_docs_map(catalog: DocsCatalog) -> None:
    docs_readme = docs_root() / "README.md"
    docs_index = docs_root() / "index.md"
    map_content = _render_docs_map(catalog)

    readme_text = docs_readme.read_text(encoding="utf-8")
    readme_text = _replace_marker_block(readme_text, DOCS_MAP_MARKER, map_content)
    _write_if_changed(docs_readme, readme_text)

    index_text = docs_index.read_text(encoding="utf-8")
    index_text = _replace_marker_block(index_text, DOCS_MAP_MARKER, map_content)
    _write_if_changed(docs_index, index_text)


def sync_runbook_steps() -> None:
    runbook_steps_doc = docs_root() / "reference" / "runbook_steps.md"
    content = _render_runbook_steps()
    text = runbook_steps_doc.read_text(encoding="utf-8")
    text = _replace_marker_block(text, RUNBOOK_STEPS_MARKER, content)
    _write_if_changed(runbook_steps_doc, text)


def sync_all() -> None:
    catalog = load_docs_catalog()
    sync_docs_map(catalog)
    sync_runbook_steps()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync Cruncher docs map and generated runbook references.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero when generated docs are out of date.")
    args = parser.parse_args(argv)

    if args.check:
        catalog = load_docs_catalog()
        docs_readme = docs_root() / "README.md"
        docs_index = docs_root() / "index.md"
        runbook_steps_doc = docs_root() / "reference" / "runbook_steps.md"

        map_content = _render_docs_map(catalog)

        readme_expected = _replace_marker_block(docs_readme.read_text(encoding="utf-8"), DOCS_MAP_MARKER, map_content)
        index_expected = _replace_marker_block(docs_index.read_text(encoding="utf-8"), DOCS_MAP_MARKER, map_content)
        runbook_expected = _replace_marker_block(
            runbook_steps_doc.read_text(encoding="utf-8"),
            RUNBOOK_STEPS_MARKER,
            _render_runbook_steps(),
        )
        mismatches: list[str] = []
        if docs_readme.read_text(encoding="utf-8") != readme_expected:
            mismatches.append(str(docs_readme))
        if docs_index.read_text(encoding="utf-8") != index_expected:
            mismatches.append(str(docs_index))
        if runbook_steps_doc.read_text(encoding="utf-8") != runbook_expected:
            mismatches.append(str(runbook_steps_doc))
        if mismatches:
            print("Generated docs are out of date:")
            for item in mismatches:
                print(f"- {item}")
            return 1
        return 0

    sync_all()
    print("Cruncher docs map and runbook-step references are synced.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
