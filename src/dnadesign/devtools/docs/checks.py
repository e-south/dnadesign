"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/checks.py

Validates docs markdown naming, local links, and public interface doc contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from dnadesign.devtools.ci.changes import discover_repo_tools
from dnadesign.devtools.docs.freshness import collect_changed_doc_dates, verification_change_issue
from dnadesign.devtools.docs.metadata import LAST_VERIFIED_PATTERN, OWNER_PATTERN, SOR_MARKDOWN_FILES
from dnadesign.ops.catalog import (
    CatalogProcedureEntry,
    load_runbook_catalog,
    render_catalog_procedure_section,
    render_catalog_tool_source_section,
    resolve_catalog_doc_path,
    resolve_registry_metadata_path_for_doc_path,
)
from dnadesign.ops.runbooks import (
    PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR,
    REPO_TRANSIENT_OPERATIONAL_DIR_NAMES,
)
from dnadesign.ops.status import list_status_kind_specs_for_repo

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
README_TOOL_LINK_PATTERN = re.compile(r"\[\*\*(?P<tool>[a-z0-9_-]+)\*\*\]\((?P<link>[^)]+)\)")
README_COVERAGE_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((?P<link>[^)]+)\)")
TOOL_README_BANNER_PATTERN = re.compile(r"!\[[^\]]*banner[^\]]*\]\((?P<link>[^)]+)\)", flags=re.IGNORECASE)
TOOL_README_BANNER_DIMENSION_PATTERN = re.compile(
    r"<svg[^>]*\bwidth=\"1200\"[^>]*\bheight=\"180\"[^>]*\bviewBox=\"0 0 1200 180\"",
    flags=re.IGNORECASE,
)
TOOL_README_SELF_REFERENTIAL_INTRO_PATTERN = re.compile(
    r"\b[A-Za-z0-9_-]+\s+is\s+(?:the|a)\s+[^.]*\b(?:package|tool|layer)\b[^.]*\b(?:in|for)\s+`?dnadesign`?",
    flags=re.IGNORECASE,
)
README_TOOL_CATALOG_EXCLUDED_TOOLS = {"studies"}
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOOL_README_MAX_LINES = 35
TOOL_README_TOP_LINK_SCAN_LINES = 80
RUNBOOK_DEMO_SHELL_LANGS = {"bash", "sh", "zsh"}
RUNBOOK_DEMO_YAML_LANGS = {"yaml", "yml"}
RUNBOOK_DEMO_HEREDOC_PATTERN = re.compile(r"""<<[-~]?\s*['"]?([A-Za-z_][A-Za-z0-9_]*)['"]?""")
RUNBOOK_DEMO_YAML_VALUE_PATTERN = re.compile(r"^\s*(?:-\s*)?[A-Za-z0-9_\-]+\s*:\s*.+$")
RUNBOOK_DEMO_CONTROL_PREFIXES = (
    "if ",
    "then",
    "else",
    "elif ",
    "fi",
    "for ",
    "while ",
    "do",
    "done",
    "case ",
    "esac",
    "function ",
    "return ",
)
ROOT_MARKDOWN_FILES = (
    "README.md",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "DESIGN.md",
    "SECURITY.md",
    "RELIABILITY.md",
    "PLANS.md",
    "QUALITY_SCORE.md",
)
INDEX_MARKDOWN_FILES = (
    "docs/README.md",
    "docs/setup/README.md",
    "docs/notebooks/README.md",
    "docs/runbooks/README.md",
    "docs/architecture/README.md",
    "docs/architecture/decisions/README.md",
    "docs/security/README.md",
    "docs/reliability/README.md",
    "docs/quality/README.md",
    "docs/exec-plans/README.md",
    "docs/templates/README.md",
    "docs/dev/README.md",
    "docs/bu-scc/README.md",
    "docs/notify/README.md",
    "docs/studies/reference/README.md",
)
RUNBOOK_MARKDOWN_FILES = (
    "docs/setup/installation.md",
    "docs/setup/dependencies.md",
    "docs/notebooks/README.md",
    "docs/notebooks/marimo-reference.md",
    "docs/operations/README.md",
    "docs/operations/orchestration/runbooks.md",
    "docs/bu-scc/setup/quickstart.md",
    "docs/bu-scc/setup/install.md",
    "docs/bu-scc/runbooks/batch-notify.md",
    "docs/notify/usr-events.md",
)
TYPE_PATTERN = re.compile(r"^\*\*Type:\*\*\s*(.+?)\s*$", re.MULTILINE)
PLANE_PATTERN = re.compile(r"^\*\*Plane:\*\*\s*(.+?)\s*$", re.MULTILINE)
OWNER_BOUNDARY_PATTERN = re.compile(r"^\*\*Owner-boundary:\*\*\s*(.+?)\s*$", re.MULTILINE)
ENTRY_ARTIFACT_PATTERN = re.compile(r"^\*\*Entry artifact:\*\*\s*(.+?)\s*$", re.MULTILINE)
EXIT_ARTIFACT_PATTERN = re.compile(r"^\*\*Exit artifact:\*\*\s*(.+?)\s*$", re.MULTILINE)
REGISTRY_ID_METADATA_PATTERN = re.compile(r"^\*\*Registry-id:\*\*\s*(.+?)\s*$", re.MULTILINE)
SUMMARY_METADATA_PATTERN = re.compile(r"^\*\*Summary:\*\*\s*(.+?)\s*$", re.MULTILINE)
EXECUTION_KIND_METADATA_PATTERN = re.compile(r"^\*\*Execution-kind:\*\*\s*(.+?)\s*$", re.MULTILINE)
STATUS_KIND_METADATA_PATTERN = re.compile(r"^\*\*Status-kind:\*\*\s*(.+?)\s*$", re.MULTILINE)
PROGRESS_SURFACE_GLOSSARY_ROW_PATTERN = re.compile(r"^\|\s*`(?P<kind>[^`]+)`\s*\|")
STATUS_PATTERN = re.compile(r"^\*\*Status:\*\*\s*(.+?)\s*$", re.MULTILINE)
CREATED_PATTERN = re.compile(r"^\*\*Created:\*\*\s*(.+?)\s*$", re.MULTILINE)
SECTION_HEADING_PATTERN = re.compile(r"^#{2,6}\s+(.+?)\s*$", re.MULTILINE)
CHECKLIST_ITEM_PATTERN = re.compile(r"^\s*-\s*\[[ xX]\]\s+", re.MULTILINE)
PROGRESS_ITEM_TIMESTAMP_PATTERN = re.compile(r"^\s*-\s*\[[ xX]\]\s+\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}Z\)\s+.+$")
_EXEC_PLAN_STATUSES = {"active", "paused", "completed"}
_EXEC_PLAN_REQUIRED_SECTIONS = (
    "Purpose / Big Picture",
    "Progress",
    "Surprises & Discoveries",
    "Decision Log",
    "Outcomes & Retrospective",
    "Context and Orientation",
    "Plan of Work",
    "Concrete Steps",
    "Validation and Acceptance",
)
PUBLIC_INTERFACE_DOC_PATHS = (
    "docs/README.md",
    "docs/dev/README.md",
    "docs/runbooks/README.md",
    "docs/studies/README.md",
    "src/dnadesign/cruncher/docs/demos",
    "src/dnadesign/cruncher/docs/reference/cli.md",
    "src/dnadesign/cruncher/workspaces",
    "src/dnadesign/densegen/README.md",
    "src/dnadesign/densegen/docs/howto",
    "src/dnadesign/densegen/docs/tutorials",
    "src/dnadesign/densegen/workspaces/README.md",
    "src/dnadesign/ops/README.md",
    "src/dnadesign/ops/docs",
    "src/dnadesign/studies/README.md",
)
ABSOLUTE_DOC_PATH_TOKENS = ("/Users/", "/private/", "/tmp/", "/home/", "/var/", "C:\\")
INTERNAL_SOURCE_INREACH_PATTERN = re.compile(r"(?:dnadesign\.[a-z0-9_]+\.src\.|src/dnadesign/[a-z0-9_-]+/src/)")
ENTRYPOINT_MARKDOWN_FILES = ("README.md", "docs/README.md")
ENTRYPOINT_LOCAL_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./-])(?P<path>(?:\.\./|\.\/)?(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+)(?![A-Za-z0-9_./-])"
)
AGENTS_CODE_SPAN_PATTERN = re.compile(r"`([^`\n]+)`")
AGENTS_REPO_RELATIVE_PREFIXES = ("src/", "docs/", ".agents/", ".github/", "tests/", "scripts/")
AGENTS_ROOT_FILENAMES = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "DESIGN.md",
    "README.md",
    "RELIABILITY.md",
    "SECURITY.md",
    "PLANS.md",
    "QUALITY_SCORE.md",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
    "pixi.toml",
    "pixi.lock",
}
AGENTS_PATH_LITERAL_EXCEPTIONS = {"./scripts/agent-verify", "scripts/agent-verify"}
AGENTS_NEGATIVE_LINE_MARKERS = ("do not add", "does not ship", "does not exist")
REPO_LOCAL_SKILLS_DIR = ".agents/skills"
REPO_LOCAL_SKILL_DESCRIPTION_MAX_CHARS = 220
DENSEGEN_DOC_LANGUAGE_PATHS = (
    "src/dnadesign/densegen/README.md",
    "src/dnadesign/densegen/AGENTS.md",
    "src/dnadesign/densegen/docs",
    "src/dnadesign/densegen/workspaces",
)
DENSEGEN_DISALLOWED_TERM_PATTERN = re.compile(r"\bcanonical\b", flags=re.IGNORECASE)
CONSTRUCT_OPERATOR_DOC_PATHS = (
    "src/dnadesign/usr/docs/operations",
    "docs/studies",
    "src/dnadesign/notify/docs/reference/command-contracts.md",
)
CONSTRUCT_LEGACY_OPERATOR_PATTERNS = (
    re.compile(r'data\["job"\]\["input"\]\["dataset"\]'),
    re.compile(r'data\["job"\]\["input"\]\["root"\]'),
    re.compile(r'data\["job"\]\["output"\]\["dataset"\]'),
    re.compile(r'data\["job"\]\["output"\]\["root"\]'),
    re.compile(r'project\["input_dataset"\]'),
    re.compile(r'project\["output_dataset"\]'),
    re.compile(r"`input\.dataset`"),
    re.compile(r"`input\.root`"),
    re.compile(r"`output\.dataset`"),
    re.compile(r"`output\.root`"),
)
LEGACY_CONTRACT_SURFACE_DOC_PATTERNS = (
    re.compile(r"\bdnadesign\._contracts\b"),
    re.compile(r"\bdnadesign\.usr_roots\b"),
    re.compile(r"src/dnadesign/_contracts\b"),
    re.compile(r"src/dnadesign/usr_roots\.py\b"),
    re.compile(r"src/dnadesign/usr/src/roots\.py\b"),
    re.compile(r"src/dnadesign/ops/orchestrator/contracts\.py\b"),
)
STUDY_RECORD_REQUIRED_FILES = (
    "record/campaign.yaml",
    "record/datasets.yaml",
    "record/status.md",
    "operations/ops.study.yaml",
)
STUDY_OPS_CONTRACT_PART_KEYS = {
    "lifecycle",
    "phases",
    "tracks",
    "artifacts",
    "execution_surfaces",
    "snapshot",
    "preflight",
}
STUDY_OPS_CONTRACT_PARTS_DIR = "contract"
STUDY_OPS_CONTRACT_PART_MAX_LINES = 180
STUDY_README_FRONTMATTER_REQUIRED_KEYS = {
    "doc_id",
    "surface",
    "study_id",
    "owner",
    "last_verified",
}
STUDY_RUNTIME_PIPELINE_REF = "manifest:operations/runtime/command-groups/pipeline.yaml"
STUDY_LEGACY_PIPELINE_REFS = {
    "manifest:operations/pipeline.yaml",
    "manifest:operations/runtime/pipeline.yaml",
    "operations/pipeline.yaml",
    "operations/runtime/pipeline.yaml",
}
STUDY_RECORD_REQUIRED_READMES = ("docs/studies/README.md",)
STUDY_RECORD_ROUTER_FILES = (
    "AGENTS.md",
    "src/dnadesign/usr/AGENTS.md",
)
STUDY_STATUS_SURFACE_SEMANTICS_DOC_PATHS = (
    "ARCHITECTURE.md",
    "docs/README.md",
    "docs/studies/README.md",
    "docs/studies/reference/study-status-ops-surfaces.md",
    "docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md",
    "docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md",
    "docs/studies/stress_ethanol_cipro_growth/routes/README.md",
    "docs/studies/stress_ethanol_cipro_growth/routes/analysis/latentdna.md",
    "docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md",
    "docs/studies/retron_hairpin_design/operations/catalog/contracts/status.md",
    "docs/studies/retron_hairpin_design/operations/catalog/contracts/preflight.md",
)
LEGACY_STUDY_STATUS_SURFACE_TERMS = (
    "Study status adapters",
    "study-status adapter",
    "status adapter policy",
    "study-family policy",
    "family routing resolves",
    "src/dnadesign/studies/families/",
    "Study-family adapters",
    "family-specific execution taxonomy",
    "family-owned status",
    "`family`, and `record_root`",
    "declare `family` and `record_root`",
    "promoter-family code",
    "promoter-family adapter",
)
ACTIVE_STUDY_INDEX_PATH = "docs/studies/index.yaml"
LEGACY_STUDY_INDEX_PATH = "docs/studies/promoter/index.yaml"
LEGACY_STUDY_RECORD_PREFIX = "docs/studies/promoter/"
ACTIVE_SHARED_USR_DATASET_ID_NUDGE = (
    "Active shared USR dataset IDs must be flat owner-first IDs like "
    "'densegen_prom_eth_cip_source'; use root_kind, owner_tool, overlays, and "
    "study metadata for provenance. archived/ is the only special top-level bucket."
)
SHARED_USR_DATASETS_ROOT = "src/dnadesign/usr/datasets"
SHARED_USR_DATASET_LAYOUT_NUDGE = (
    "Shared repo USR dataset roots must be flat under src/dnadesign/usr/datasets; "
    "move nested roots to owner-first ids like 'densegen_demo_sampling_baseline'. "
    "archived/ is the only special top-level bucket."
)
OPS_OPERATIONAL_WORKFLOW_IDS = {
    "densegen_batch_submit",
    "densegen_batch_with_notify",
    "infer_batch_submit",
    "infer_batch_with_notify",
}
CROSS_TOOL_DOC_METADATA_CONTRACTS: dict[str, dict[str, str]] = {
    "docs/operations/README.md": {
        "type": "route",
        "plane": "control-plane",
        "owner_boundary": "ops",
    },
    "docs/operations/orchestration/runbooks.md": {
        "type": "runbook",
        "plane": "control-plane",
        "owner_boundary": "ops",
    },
    "src/dnadesign/usr/docs/operations/README.md": {
        "type": "route",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/routes/workflow-map.md": {
        "type": "route",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/sync/README.md": {
        "type": "route",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.md": {
        "type": "runbook",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/assembly/multi-source-shared-dataset.md": {
        "type": "runbook",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/sync/chained-densegen-infer-runbook.md": {
        "type": "runbook",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md": {
        "type": "runbook",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/assembly/permuter-construct-infer-shared-dataset.md": {
        "type": "runbook",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "src/dnadesign/usr/docs/operations/promoter/characterization-feature-matrix.md": {
        "type": "runbook",
        "plane": "data-plane",
        "owner_boundary": "usr",
    },
    "docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md": {
        "type": "contract",
        "plane": "data-plane",
        "owner_boundary": "studies",
    },
    "docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md": {
        "type": "contract",
        "plane": "data-plane",
        "owner_boundary": "studies",
    },
    "docs/studies/retron_hairpin_design/operations/catalog/contracts/status.md": {
        "type": "contract",
        "plane": "data-plane",
        "owner_boundary": "studies",
    },
    "docs/studies/retron_hairpin_design/operations/catalog/contracts/preflight.md": {
        "type": "contract",
        "plane": "data-plane",
        "owner_boundary": "studies",
    },
    "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md": {
        "type": "workflow",
        "plane": "downstream-tool",
        "owner_boundary": "cluster",
    },
    "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md": {
        "type": "workflow",
        "plane": "downstream-tool",
        "owner_boundary": "opal",
    },
}
_CROSS_TOOL_DOC_ALLOWED_TYPES = {"contract", "route", "runbook", "workflow"}
_CROSS_TOOL_DOC_ALLOWED_PLANES = {"control-plane", "data-plane", "downstream-tool"}
_RUNBOOK_CATALOG_METADATA_TYPES = {"contract", "runbook", "workflow"}
_REGISTRY_ID_VALUE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)+$")
_METADATA_TOKEN_VALUE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*(?:-[a-z0-9]+)*$")
RUNBOOK_CATALOG_DOC_PATH = "docs/runbooks/README.md"
RUNBOOK_STATUS_GLOSSARY_HEADING = "### Status views"
OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES = (
    PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR,
    Path("docs/templates"),
)
OPS_OPERATIONAL_RUNBOOK_FALLBACK_SCAN_ROOTS = (
    PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR,
    Path("docs/templates"),
)
TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES = REPO_TRANSIENT_OPERATIONAL_DIR_NAMES
DISALLOWED_SHARED_UTILS_PATHS = (Path("src/dnadesign/utils"),)
DISALLOWED_REPO_ROOT_OUTPUT_DIR_NAMES = ("outputs",)
OVERLAY_GUARD_DOC_PATHS = (
    "docs/operations/orchestration/runbooks.md",
    "docs/bu-scc/jobs/README.md",
    "src/dnadesign/ops/README.md",
)
OPS_DEPRECATED_SEMANTICS_DOC_PATHS = (
    "docs/operations/README.md",
    "docs/operations/orchestration/runbooks.md",
    "docs/studies/README.md",
    "docs/studies/stress_ethanol_cipro_growth/record/status.md",
    "src/dnadesign/ops/README.md",
    "docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md",
)
STUDY_EXECUTION_SOURCE_DOC_PATHS = (
    "docs/studies/README.md",
    "docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md",
)
STALE_OVERLAY_GUARD_TERMS = (
    "densegen-overlay-guard",
    "densegen.overlay_guard.namespace",
)
OPS_DEPRECATED_SEMANTICS_TERMS = (
    "precedent",
    "precedents",
    "with_notify_slack",
    "infer_validate_config",
    "infer_local_runtime",
    "infer_dry_run",
    "notify_profile_doctor",
    "notify_resolve_events",
    "details.setup_command",
)
PIPELINE_ONLY_EXECUTION_SOURCE_PATTERN = re.compile(
    r"pipeline\.yaml[\s\S]{0,120}only (?:valid )?source",
    flags=re.IGNORECASE,
)
PACKAGED_RUNBOOK_DURATION_SUFFIX_PATTERN = re.compile(r"_(?:\d+)(?:h|hr|hrs|hour|hours)$", re.IGNORECASE)
OPERATIONAL_RUNBOOK_SCAN_PRUNE_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "_archive",
    "_auxiliary",
    "_derived",
    "_snapshots",
    "archived",
    "batch_results",
    "build",
    "dist",
    "prototype",
    "prototypes",
    "runs",
    "venv",
}


def _collect_markdown_files(repo_root: Path) -> tuple[list[Path], list[Path]]:
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        raise FileNotFoundError("docs/ directory is missing")

    docs_md_files = sorted(docs_root.rglob("*.md"))
    tool_docs_md_files = _collect_tool_docs_markdown_files(repo_root)
    tool_readme_md_files = _collect_tool_readme_markdown_files(repo_root)
    all_md_files = list(docs_md_files)
    all_md_files.extend(tool_docs_md_files)
    all_md_files.extend(tool_readme_md_files)
    for name in ROOT_MARKDOWN_FILES:
        path = repo_root / name
        if path.exists():
            all_md_files.append(path)
    deduped = sorted(set(all_md_files))
    return docs_md_files, deduped


def _collect_tool_docs_markdown_files(repo_root: Path) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    tool_docs: set[Path] = set()
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        docs_root = src_root / tool_name / "docs"
        if not docs_root.exists():
            continue
        for path in docs_root.rglob("*.md"):
            tool_docs.add(path)
    return sorted(tool_docs)


def _collect_tool_readme_markdown_files(repo_root: Path) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    files: list[Path] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        readme_path = src_root / tool_name / "README.md"
        if readme_path.exists():
            files.append(readme_path)
    return files


def _collect_markdown_files_from_relative_paths(repo_root: Path, *, relative_paths: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for rel in relative_paths:
        target = repo_root / rel
        if not target.exists():
            continue
        if target.is_file() and target.suffix == ".md":
            files.add(target)
            continue
        if target.is_dir():
            for path in target.rglob("*.md"):
                files.add(path)
    return sorted(files)


def _find_bad_doc_names(docs_md_files: list[Path]) -> list[Path]:
    return [path for path in docs_md_files if "_" in path.name]


def _find_broken_links(md_files: list[Path], *, repo_root: Path | None = None) -> list[tuple[Path, str]]:
    broken: list[tuple[Path, str]] = []
    anchor_cache: dict[Path, set[str]] = {}
    resolved_repo_root = repo_root.expanduser().resolve() if repo_root is not None else None
    for src in md_files:
        text = _markdown_text_without_fenced_code(src.read_text(encoding="utf-8"))
        for raw in LINK_PATTERN.findall(text):
            link = raw.strip().split()[0]
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target_rel, anchor = (link.split("#", 1) + [""])[:2]
            if not target_rel:
                target = src.resolve()
            else:
                target = (src.parent / target_rel).resolve()
            if resolved_repo_root is not None:
                try:
                    target.relative_to(resolved_repo_root)
                except ValueError:
                    broken.append((src, f"{link} (local link escapes repository)"))
                    continue
            if not target.exists():
                broken.append((src, link))
                continue
            if anchor and target.suffix == ".md":
                if target not in anchor_cache:
                    anchor_cache[target] = _collect_markdown_anchors(target)
                if anchor not in anchor_cache[target]:
                    broken.append((src, f"{link} (missing anchor '{anchor}')"))
    return broken


def _markdown_text_without_fenced_code(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    fence_marker: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        marker = None
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"

        if marker is not None:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            lines.append("")
            continue

        lines.append("" if in_fence else raw_line)
    return "\n".join(lines)


def _collect_markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    slug_counts: dict[str, int] = {}
    for _, _, heading_text in _collect_markdown_headings_outside_fences(path):
        slug = _slugify_markdown_heading(heading_text)
        if not slug:
            continue
        count = slug_counts.get(slug, 0)
        slug_counts[slug] = count + 1
        if count == 0:
            anchors.add(slug)
        else:
            anchors.add(f"{slug}-{count}")
    return anchors


def _slugify_markdown_heading(value: str) -> str:
    chars: list[str] = []
    for char in value.strip().lower():
        if char.isalnum() or char in {" ", "-", "_"}:
            chars.append(char)
    slug = "".join(chars).replace(" ", "-")
    return slug.strip("-")


def _extract_level2_section_lines(text: str, heading: str) -> list[str]:
    section_lines: list[str] = []
    in_section = False
    target = f"## {heading}"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == target:
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section:
            section_lines.append(line)
    return section_lines


def _readme_tool_table_rows(text: str) -> list[list[str]]:
    section_lines = _extract_level2_section_lines(text, "Available tools")
    if not section_lines:
        return []
    rows: list[list[str]] = []
    for line in section_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if len(cells) >= 2 and cells[0].lower() == "tool" and cells[1].lower() == "description":
            continue
        if set(stripped.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue
        rows.append(cells)
    return rows


def _normalize_relative_markdown_path(value: str) -> str:
    return str(Path(value).as_posix().lstrip("./"))


def _is_valid_codecov_component_link(*, tool_name: str, link: str) -> bool:
    parsed = urlparse(link)
    if parsed.scheme != "https":
        return False
    if parsed.netloc not in {"codecov.io", "www.codecov.io", "app.codecov.io"}:
        return False
    if not parsed.path.startswith("/gh/"):
        return False
    component_values = parse_qs(parsed.query).get("component")
    if component_values is None:
        return False
    return any(value.strip() == tool_name for value in component_values)


def _find_readme_tool_catalog_issues(repo_root: Path) -> list[str]:
    readme_path = repo_root / "README.md"
    src_root = repo_root / "src" / "dnadesign"
    if not readme_path.exists() or not src_root.exists():
        return []

    repo_tools = discover_repo_tools(repo_root=repo_root) - README_TOOL_CATALOG_EXCLUDED_TOOLS
    if not repo_tools:
        return []

    readme_text = readme_path.read_text(encoding="utf-8")
    rows = _readme_tool_table_rows(readme_text)
    if not rows:
        return [f"{readme_path}: section '## Available tools' must include a markdown tool table."]

    issues: list[str] = []
    declared_tools: set[str] = set()
    for row in rows:
        if len(row) < 3:
            issues.append(f"{readme_path}: tool table rows must include Tool, Description, and Coverage columns.")
            continue

        tool_cell = row[0]
        coverage_cell = row[2]
        match = README_TOOL_LINK_PATTERN.search(tool_cell)
        if match is None:
            issues.append(f"{readme_path}: tool cell must use [**tool**](src/dnadesign/tool) format ({tool_cell}).")
            continue

        tool_name = match.group("tool")
        tool_link = match.group("link")
        if tool_name in declared_tools:
            issues.append(f"{readme_path}: duplicate tool row for '{tool_name}'.")
            continue
        declared_tools.add(tool_name)

        expected_rel = Path("src") / "dnadesign" / tool_name / "README.md"
        if _normalize_relative_markdown_path(tool_link) != expected_rel.as_posix():
            issues.append(
                f"{readme_path}: tool '{tool_name}' must link to '{expected_rel.as_posix()}' (found '{tool_link}')."
            )

        tool_readme = (repo_root / tool_link).resolve()
        if not tool_readme.exists() or not tool_readme.is_file():
            issues.append(
                f"{readme_path}: tool '{tool_name}' link target does not exist as a markdown file: {tool_link}."
            )

        coverage_match = README_COVERAGE_LINK_PATTERN.search(coverage_cell)
        if coverage_match is None:
            issues.append(
                f"{readme_path}: coverage cell for '{tool_name}' must include a markdown link "
                "to a Codecov component URL."
            )
            continue
        coverage_link = coverage_match.group("link")
        if not _is_valid_codecov_component_link(tool_name=tool_name, link=coverage_link):
            issues.append(
                f"{readme_path}: coverage link for '{tool_name}' must target Codecov with query "
                f"'component={tool_name}' (found '{coverage_link}')."
            )

    missing_tools = sorted(repo_tools - declared_tools)
    extra_tools = sorted(declared_tools - repo_tools)
    if missing_tools:
        issues.append(f"{readme_path}: missing tool rows for: {', '.join(missing_tools)}.")
    if extra_tools:
        issues.append(f"{readme_path}: unknown tool rows not found in src/dnadesign: {', '.join(extra_tools)}.")
    return issues


def _find_root_docs_entrypoint_issues(repo_root: Path) -> list[str]:
    readme_path = repo_root / "README.md"
    if not readme_path.exists():
        return []

    text = readme_path.read_text(encoding="utf-8")
    linked_targets: set[str] = set()
    for raw in LINK_PATTERN.findall(text):
        link = raw.strip().split()[0]
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target_rel = link.split("#", 1)[0].strip()
        if not target_rel:
            continue
        linked_targets.add(_normalize_relative_markdown_path(target_rel))

    issues: list[str] = []
    if "docs/README.md" not in linked_targets:
        issues.append(f"{readme_path}: root README must include a markdown link to docs/README.md.")
    return issues


def _collect_markdown_headings_outside_fences(path: Path) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = MARKDOWN_HEADING_PATTERN.match(raw_line)
        if match is None:
            continue
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        headings.append((line_no, level, heading_text))
    return headings


def _find_docs_root_heading_style_issues(repo_root: Path) -> list[str]:
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        return []

    issues: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        if "outputs" in path.relative_to(repo_root).parts:
            continue
        headings = _collect_markdown_headings_outside_fences(path)
        if not headings:
            continue

        first_line_no, first_level, _ = headings[0]
        if first_level != 2:
            issues.append(f"{path}:{first_line_no}: docs root markdown must start with '## ' heading style.")

        level2_count = sum(1 for _, level, _ in headings if level == 2)
        if level2_count > 1:
            issues.append(
                f"{path}: docs root markdown should use a single level-2 heading; use level-3+ for subsections."
            )
    return issues


def _find_deprecated_docs_entrypoint_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    deprecated_start_here = repo_root / "docs" / "start-here.md"
    if deprecated_start_here.exists():
        issues.append(f"{deprecated_start_here}: deprecated docs shim must not exist; consolidate into docs/README.md.")

    check_paths = (repo_root / "README.md", repo_root / "docs" / "README.md")
    disallowed_targets = {"docs/start-here.md", "start-here.md"}
    for path in check_paths:
        if not path.exists():
            continue
        for raw in LINK_PATTERN.findall(path.read_text(encoding="utf-8")):
            link = raw.strip().split()[0]
            target_rel = link.split("#", 1)[0].strip()
            if target_rel in disallowed_targets:
                issues.append(f"{path}: must not link to docs/start-here.md; use docs/README.md.")
    return issues


def _find_entrypoint_local_path_literal_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for relative_path in ENTRYPOINT_MARKDOWN_FILES:
        path = repo_root / relative_path
        if not path.exists():
            continue

        lines = path.read_text(encoding="utf-8").splitlines()
        in_fence = False
        for line_no, line in enumerate(lines, start=1):
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            line_without_links = LINK_PATTERN.sub("", line)
            for match in ENTRYPOINT_LOCAL_PATH_PATTERN.finditer(line_without_links):
                token = match.group("path").strip("()[]{}<>.,:;!?")
                if not token:
                    continue
                if token.startswith(("http://", "https://", "mailto:", "#")):
                    continue

                repo_target = (repo_root / token).resolve()
                relative_target = (path.parent / token).resolve()
                if not repo_target.exists() and not relative_target.exists():
                    continue

                issues.append(
                    f"{path}:{line_no}: local path literal '{token}' should be a markdown hyperlink for navigation."
                )
    return issues


def _find_agents_path_reference_issues(repo_root: Path) -> list[str]:
    resolved_repo_root = repo_root.expanduser().resolve()
    issues: list[str] = []
    skipped_dir_names = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".worktrees",
        "__pycache__",
    }
    for agents_path in sorted(resolved_repo_root.rglob("AGENTS.md")):
        relative_agents_path = agents_path.relative_to(resolved_repo_root)
        if any(part in skipped_dir_names for part in relative_agents_path.parts):
            continue
        for line_no, line in enumerate(agents_path.read_text(encoding="utf-8").splitlines(), start=1):
            line_lower = line.lower()
            if any(marker in line_lower for marker in AGENTS_NEGATIVE_LINE_MARKERS):
                continue
            for raw_value in AGENTS_CODE_SPAN_PATTERN.findall(line):
                target_literal = raw_value.strip().strip("()[]{}<>,:;!?")
                if not target_literal or target_literal in AGENTS_PATH_LITERAL_EXCEPTIONS:
                    continue
                if _should_skip_agents_path_literal(target_literal):
                    continue

                target_path = _resolve_agents_path_literal(
                    target_literal,
                    agents_path=agents_path,
                    repo_root=resolved_repo_root,
                )
                if target_path is None:
                    continue
                if not target_path.exists():
                    issues.append(f"{agents_path}:{line_no}: referenced path does not exist: `{target_literal}`")
    return issues


def _should_skip_agents_path_literal(value: str) -> bool:
    if value.startswith(("http://", "https://", "mailto:", "#", "/Users/", "/private/", "/tmp/", "/home/", "/var/")):
        return True
    if any(char.isspace() for char in value):
        return True
    if any(char in value for char in "*?[]{}<>$|;&="):
        return True
    if "://" in value or ":" in value:
        return True

    path_part = value.split("#", 1)[0]
    if path_part.startswith(AGENTS_REPO_RELATIVE_PREFIXES):
        return False
    if path_part in AGENTS_ROOT_FILENAMES:
        return False
    if path_part.startswith("./") and path_part[2:] in AGENTS_ROOT_FILENAMES:
        return False
    return True


def _resolve_agents_path_literal(value: str, *, agents_path: Path, repo_root: Path) -> Path | None:
    path_part = value.split("#", 1)[0]
    if path_part.startswith("./") and path_part[2:] in AGENTS_ROOT_FILENAMES:
        path_part = path_part[2:]
    if path_part.startswith(AGENTS_REPO_RELATIVE_PREFIXES):
        repo_target = repo_root / path_part
        relative_target = agents_path.parent / path_part
        target = relative_target if relative_target.exists() else repo_target
    elif path_part in AGENTS_ROOT_FILENAMES:
        target = repo_root / path_part
    else:
        target = agents_path.parent / path_part
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(repo_root)
    except ValueError:
        return None
    return resolved_target


def _find_densegen_disallowed_term_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    targets = _collect_markdown_files_from_relative_paths(repo_root, relative_paths=DENSEGEN_DOC_LANGUAGE_PATHS)
    for path in targets:
        content = path.read_text(encoding="utf-8")
        match = DENSEGEN_DISALLOWED_TERM_PATTERN.search(content)
        if match is None:
            continue
        line_no = content[: match.start()].count("\n") + 1
        issues.append(f"{path}:{line_no}: term '{match.group(0)}' is not allowed in DenseGen docs.")
    return issues


def _find_construct_legacy_operator_doc_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    targets = _collect_markdown_files_from_relative_paths(repo_root, relative_paths=CONSTRUCT_OPERATOR_DOC_PATHS)
    for path in targets:
        content = path.read_text(encoding="utf-8")
        for pattern in CONSTRUCT_LEGACY_OPERATOR_PATTERNS:
            match = pattern.search(content)
            if match is None:
                continue
            line_no = content[: match.start()].count("\n") + 1
            issues.append(
                f"{path}:{line_no}: deprecated construct flat-key contract "
                f"'{match.group(0)}' is not allowed in operator docs."
            )
    return issues


def _find_legacy_contract_surface_doc_issues(repo_root: Path) -> list[str]:
    try:
        _docs_md_files, all_md_files = _collect_markdown_files(repo_root)
    except FileNotFoundError:
        return []

    issues: list[str] = []
    for path in all_md_files:
        content = path.read_text(encoding="utf-8")
        for pattern in LEGACY_CONTRACT_SURFACE_DOC_PATTERNS:
            match = pattern.search(content)
            if match is None:
                continue
            line_no = content[: match.start()].count("\n") + 1
            issues.append(
                f"{path}:{line_no}: legacy repo-root contract surface reference "
                f"'{match.group(0)}' is not allowed in docs."
            )
    return issues


def _find_codecov_component_issues(repo_root: Path) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []
    repo_tools = discover_repo_tools(repo_root=repo_root)
    if not repo_tools:
        return []

    codecov_path = repo_root / "codecov.yml"
    if not codecov_path.exists():
        return [f"{codecov_path}: missing Codecov configuration file."]

    try:
        config = yaml.safe_load(codecov_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{codecov_path}: unable to parse YAML ({exc})."]

    if not isinstance(config, dict):
        return [f"{codecov_path}: expected a top-level YAML mapping."]

    component_management = config.get("component_management")
    if not isinstance(component_management, dict):
        return [f"{codecov_path}: missing 'component_management' mapping."]

    default_rules = component_management.get("default_rules")
    if not isinstance(default_rules, dict):
        return [f"{codecov_path}: missing component_management.default_rules mapping."]
    statuses = default_rules.get("statuses")
    if not isinstance(statuses, list):
        return [f"{codecov_path}: missing component_management.default_rules.statuses list."]

    has_required_status = False
    for status in statuses:
        if not isinstance(status, dict):
            continue
        if (
            status.get("type") == "project"
            and status.get("target") == "auto"
            and status.get("if_ci_failed") == "error"
            and status.get("if_not_found") == "failure"
        ):
            has_required_status = True
            break
    if not has_required_status:
        return [
            f"{codecov_path}: component_management.default_rules.statuses must include a "
            "project status with target=auto, if_ci_failed=error, if_not_found=failure."
        ]

    individual_components = component_management.get("individual_components")
    if not isinstance(individual_components, list):
        return [f"{codecov_path}: missing component_management.individual_components list."]

    component_ids: set[str] = set()
    issues: list[str] = []

    for component in individual_components:
        if not isinstance(component, dict):
            issues.append(f"{codecov_path}: each individual component must be a mapping.")
            continue

        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            issues.append(f"{codecov_path}: each component must define a non-empty component_id.")
            continue
        if component_id in component_ids:
            issues.append(f"{codecov_path}: duplicate component_id '{component_id}'.")
            continue
        component_ids.add(component_id)

        paths = component.get("paths")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            issues.append(f"{codecov_path}: component '{component_id}' must define 'paths' as a list of strings.")
            continue
        expected_path = f"src/dnadesign/{component_id}/**"
        if expected_path not in paths:
            issues.append(f"{codecov_path}: component '{component_id}' must include path '{expected_path}'.")

    missing_components = sorted(repo_tools - component_ids)
    extra_components = sorted(component_ids - repo_tools)
    if missing_components:
        issues.append(f"{codecov_path}: missing component_id entries for: {', '.join(missing_components)}.")
    if extra_components:
        issues.append(
            f"{codecov_path}: unknown component_id entries not found in src/dnadesign: {', '.join(extra_components)}."
        )

    return issues


def _extract_metadata_field(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_iso_date(value: str, *, field_name: str, path: Path) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{path}: {field_name} must use YYYY-MM-DD.") from exc


def _find_sor_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    today = dt.date.today()
    changed_dates = changed_doc_dates or {}
    issues: list[str] = []

    for name in SOR_MARKDOWN_FILES:
        path = repo_root / name
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        doc_type = _extract_metadata_field(text, TYPE_PATTERN)
        if doc_type is None:
            issues.append(f"{path}: missing '**Type:**' metadata field.")
            continue
        if doc_type != "system-of-record":
            issues.append(f"{path}: '**Type:**' must be exactly 'system-of-record'.")
            continue

        owner = _extract_metadata_field(text, OWNER_PATTERN)
        if owner is None:
            issues.append(f"{path}: missing '**Owner:**' metadata field.")
            continue
        if not owner:
            issues.append(f"{path}: '**Owner:**' must not be empty.")

        last_verified_raw = _extract_metadata_field(text, LAST_VERIFIED_PATTERN)
        if last_verified_raw is None:
            issues.append(f"{path}: missing '**Last verified:**' metadata field.")
            continue
        if not last_verified_raw:
            issues.append(f"{path}: '**Last verified:**' must not be empty.")
            continue

        try:
            last_verified = _parse_iso_date(last_verified_raw, field_name="Last verified", path=path)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        if last_verified > today:
            issues.append(f"{path}: Last verified date cannot be in the future ({last_verified.isoformat()}).")
            continue

        change_issue = verification_change_issue(
            repo_root=repo_root,
            path=path,
            last_verified=last_verified,
            changed_doc_dates=changed_dates,
        )
        if change_issue is not None:
            issues.append(change_issue)

    return issues


def _find_index_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    return _find_owner_last_verified_metadata_issues(
        repo_root,
        relative_paths=INDEX_MARKDOWN_FILES,
        changed_doc_dates=changed_doc_dates,
    )


def _find_runbook_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    return _find_owner_last_verified_metadata_issues(
        repo_root,
        relative_paths=RUNBOOK_MARKDOWN_FILES,
        changed_doc_dates=changed_doc_dates,
    )


def _find_tool_docs_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    tool_docs = _collect_tool_docs_markdown_files(repo_root)
    return _find_owner_last_verified_metadata_issues_for_files(
        repo_root=repo_root,
        paths=tool_docs,
        changed_doc_dates=changed_doc_dates,
    )


def _find_owner_last_verified_metadata_issues(
    repo_root: Path,
    *,
    relative_paths: tuple[str, ...],
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    files = [repo_root / relative_path for relative_path in relative_paths]
    return _find_owner_last_verified_metadata_issues_for_files(
        repo_root=repo_root,
        paths=files,
        changed_doc_dates=changed_doc_dates,
    )


def _find_owner_last_verified_metadata_issues_for_files(
    *,
    repo_root: Path,
    paths: list[Path],
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    today = dt.date.today()
    changed_dates = changed_doc_dates or {}
    issues: list[str] = []

    for path in paths:
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        owner = _extract_metadata_field(text, OWNER_PATTERN)
        owner_valid = True
        if owner is None:
            issues.append(f"{path}: missing '**Owner:**' metadata field.")
            owner_valid = False
        elif not owner:
            issues.append(f"{path}: '**Owner:**' must not be empty.")

        last_verified_raw = _extract_metadata_field(text, LAST_VERIFIED_PATTERN)
        last_verified_valid = True
        if last_verified_raw is None:
            issues.append(f"{path}: missing '**Last verified:**' metadata field.")
            last_verified_valid = False
        elif not last_verified_raw:
            issues.append(f"{path}: '**Last verified:**' must not be empty.")
            last_verified_valid = False

        if not owner_valid or not last_verified_valid:
            continue

        try:
            last_verified = _parse_iso_date(last_verified_raw, field_name="Last verified", path=path)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        if last_verified > today:
            issues.append(f"{path}: Last verified date cannot be in the future ({last_verified.isoformat()}).")
            continue

        change_issue = verification_change_issue(
            repo_root=repo_root,
            path=path,
            last_verified=last_verified,
            changed_doc_dates=changed_dates,
        )
        if change_issue is not None:
            issues.append(change_issue)

    return issues


def _find_cross_tool_doc_metadata_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    registry_ids_by_path: dict[str, str] = {}
    for relative_path, contract in CROSS_TOOL_DOC_METADATA_CONTRACTS.items():
        path = repo_root / relative_path
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")

        doc_type = _extract_metadata_field(text, TYPE_PATTERN)
        if doc_type is None:
            issues.append(f"{path}: missing '**Type:**' metadata field.")
        elif doc_type not in _CROSS_TOOL_DOC_ALLOWED_TYPES:
            allowed = ", ".join(sorted(_CROSS_TOOL_DOC_ALLOWED_TYPES))
            issues.append(f"{path}: '**Type:**' must be one of: {allowed}.")
        elif doc_type != contract["type"]:
            issues.append(f"{path}: '**Type:**' must be exactly '{contract['type']}'.")

        plane = _extract_metadata_field(text, PLANE_PATTERN)
        if plane is None:
            issues.append(f"{path}: missing '**Plane:**' metadata field.")
        elif plane not in _CROSS_TOOL_DOC_ALLOWED_PLANES:
            allowed = ", ".join(sorted(_CROSS_TOOL_DOC_ALLOWED_PLANES))
            issues.append(f"{path}: '**Plane:**' must be one of: {allowed}.")
        elif plane != contract["plane"]:
            issues.append(f"{path}: '**Plane:**' must be exactly '{contract['plane']}'.")

        owner_boundary = _extract_metadata_field(text, OWNER_BOUNDARY_PATTERN)
        if owner_boundary is None:
            issues.append(f"{path}: missing '**Owner-boundary:**' metadata field.")
        elif not owner_boundary:
            issues.append(f"{path}: '**Owner-boundary:**' must not be empty.")
        elif owner_boundary != contract["owner_boundary"]:
            issues.append(f"{path}: '**Owner-boundary:**' must be exactly '{contract['owner_boundary']}'.")

        entry_artifact = _extract_metadata_field(text, ENTRY_ARTIFACT_PATTERN)
        if entry_artifact is None:
            issues.append(f"{path}: missing '**Entry artifact:**' metadata field.")
        elif not entry_artifact:
            issues.append(f"{path}: '**Entry artifact:**' must not be empty.")

        exit_artifact = _extract_metadata_field(text, EXIT_ARTIFACT_PATTERN)
        if exit_artifact is None:
            issues.append(f"{path}: missing '**Exit artifact:**' metadata field.")
        elif not exit_artifact:
            issues.append(f"{path}: '**Exit artifact:**' must not be empty.")

        if contract["type"] not in _RUNBOOK_CATALOG_METADATA_TYPES:
            continue

        registry_id = _extract_metadata_field(text, REGISTRY_ID_METADATA_PATTERN)
        if registry_id is None:
            issues.append(f"{path}: missing '**Registry-id:**' metadata field.")
        elif not registry_id:
            issues.append(f"{path}: '**Registry-id:**' must not be empty.")
        elif _REGISTRY_ID_VALUE_PATTERN.fullmatch(registry_id) is None:
            issues.append(f"{path}: '**Registry-id:**' must be dot-qualified lowercase tokens.")
        else:
            registry_ids_by_path[relative_path] = registry_id

        summary = _extract_metadata_field(text, SUMMARY_METADATA_PATTERN)
        if summary is None:
            issues.append(f"{path}: missing '**Summary:**' metadata field.")
        elif not summary:
            issues.append(f"{path}: '**Summary:**' must not be empty.")

        execution_kind = _extract_metadata_field(text, EXECUTION_KIND_METADATA_PATTERN)
        if execution_kind is None:
            issues.append(f"{path}: missing '**Execution-kind:**' metadata field.")
        elif not execution_kind:
            issues.append(f"{path}: '**Execution-kind:**' must not be empty.")
        elif _METADATA_TOKEN_VALUE_PATTERN.fullmatch(execution_kind) is None:
            issues.append(f"{path}: '**Execution-kind:**' must use lowercase slug tokens.")

        status_kind = _extract_metadata_field(text, STATUS_KIND_METADATA_PATTERN)
        if status_kind is None:
            issues.append(f"{path}: missing '**Status-kind:**' metadata field.")
        elif not status_kind:
            issues.append(f"{path}: '**Status-kind:**' must not be empty.")
        elif _METADATA_TOKEN_VALUE_PATTERN.fullmatch(status_kind) is None:
            issues.append(f"{path}: '**Status-kind:**' must use lowercase slug tokens.")

    seen_registry_ids: dict[str, str] = {}
    for relative_path, registry_id in registry_ids_by_path.items():
        existing_path = seen_registry_ids.get(registry_id)
        if existing_path is not None:
            issues.append(
                "cross-tool docs must not reuse registry ids: "
                f"{registry_id} appears in both {existing_path} and {relative_path}."
            )
            continue
        seen_registry_ids[registry_id] = relative_path

    return issues


def _find_runbook_catalog_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    resolved_repo_root = repo_root.resolve()
    catalog_paths = [
        repo_root / relative_path
        for relative_path, contract in CROSS_TOOL_DOC_METADATA_CONTRACTS.items()
        if contract["type"] in _RUNBOOK_CATALOG_METADATA_TYPES and (repo_root / relative_path).exists()
    ]
    if not catalog_paths:
        return issues

    catalog_path = repo_root / RUNBOOK_CATALOG_DOC_PATH
    if not catalog_path.exists():
        return [f"{catalog_path}: missing runbook catalog."]

    catalog_text = catalog_path.read_text(encoding="utf-8")
    if "## Runbook Catalog" not in catalog_text:
        issues.append(f"{catalog_path}: missing '## Runbook Catalog' heading.")

    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        issues.append(f"{catalog_path}: {exc}")
        return issues

    catalog_entries_by_path: dict[str, CatalogProcedureEntry] = {}
    catalog_status_kinds = {entry.status_kind for entry in catalog.procedures}
    registered_status_kinds = {spec.status_kind for spec in list_status_kind_specs_for_repo(repo_root)}
    expected_glossary_status_kinds = registered_status_kinds or catalog_status_kinds
    expected_catalog_paths: set[str] = set()
    for entry in catalog.procedures:
        resolved_path = resolve_catalog_doc_path(catalog_path=catalog.catalog_path, doc_path=entry.doc_path)
        try:
            relative_path = str(resolved_path.relative_to(resolved_repo_root))
        except ValueError:
            issues.append(
                f"{catalog_path}: registry id '{entry.registry_id}' resolves outside the repository: {entry.doc_path}."
            )
            continue
        existing_entry = catalog_entries_by_path.get(relative_path)
        if existing_entry is not None:
            issues.append(
                f"{catalog_path}: duplicate catalog procedure entry for {relative_path} "
                f"({existing_entry.registry_id}, {entry.registry_id})."
            )
            continue
        catalog_entries_by_path[relative_path] = entry

    procedure_section = _extract_markdown_section(catalog_text, heading="### Cross-tool procedures")
    if procedure_section is None:
        issues.append(f"{catalog_path}: missing '### Cross-tool procedures' section.")
    else:
        expected_section = render_catalog_procedure_section(catalog)
        if procedure_section.strip() != expected_section.strip():
            issues.append(
                f"{catalog_path}: cross-tool procedures section is stale; "
                "regenerate it with `uv run python -m dnadesign.devtools.docs.runbook_catalog`."
            )

    tool_source_section = _extract_markdown_section(catalog_text, heading="### Tool docs")
    if tool_source_section is None:
        issues.append(f"{catalog_path}: missing '### Tool docs' section.")
    else:
        expected_tool_source_section = render_catalog_tool_source_section(catalog)
        if tool_source_section.strip() != expected_tool_source_section.strip():
            issues.append(
                f"{catalog_path}: tool docs section is stale; "
                "regenerate it with `uv run python -m dnadesign.devtools.docs.runbook_catalog`."
            )

    for relative_path, contract in CROSS_TOOL_DOC_METADATA_CONTRACTS.items():
        if contract["type"] not in _RUNBOOK_CATALOG_METADATA_TYPES:
            continue
        path = repo_root / relative_path
        if not path.exists():
            continue
        expected_catalog_paths.add(relative_path)
        text = path.read_text(encoding="utf-8")
        registry_id = _extract_metadata_field(text, REGISTRY_ID_METADATA_PATTERN)
        if not registry_id:
            continue
        metadata_relative_path = resolve_registry_metadata_path_for_doc_path(relative_path)
        metadata_path = repo_root / metadata_relative_path
        if not metadata_path.exists():
            issues.append(f"{metadata_path}: missing registry metadata sidecar for {relative_path}.")
            continue
        metadata_payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        if not isinstance(metadata_payload, dict):
            issues.append(f"{metadata_path}: registry metadata must be a mapping.")
            continue

        entry = catalog_entries_by_path.get(relative_path)
        if entry is None:
            issues.append(f"{catalog_path}: missing registry id '{registry_id}' for {relative_path}.")
            continue

        expected_metadata = {
            "Registry-id": registry_id,
            "Type": _extract_metadata_field(text, TYPE_PATTERN),
            "Plane": _extract_metadata_field(text, PLANE_PATTERN),
            "Owner-boundary": _extract_metadata_field(text, OWNER_BOUNDARY_PATTERN),
            "Entry artifact": _extract_metadata_field(text, ENTRY_ARTIFACT_PATTERN),
            "Exit artifact": _extract_metadata_field(text, EXIT_ARTIFACT_PATTERN),
            "Execution-kind": _extract_metadata_field(text, EXECUTION_KIND_METADATA_PATTERN),
            "Status-kind": _extract_metadata_field(text, STATUS_KIND_METADATA_PATTERN),
            "Summary": _extract_metadata_field(text, SUMMARY_METADATA_PATTERN),
        }
        metadata_file_values = {
            "Registry-id": metadata_payload.get("registry_id"),
            "Type": metadata_payload.get("type"),
            "Plane": metadata_payload.get("plane"),
            "Owner-boundary": metadata_payload.get("owner_boundary"),
            "Entry artifact": metadata_payload.get("entry_artifact"),
            "Exit artifact": metadata_payload.get("exit_artifact"),
            "Execution-kind": metadata_payload.get("execution_kind"),
            "Status-kind": metadata_payload.get("status_kind"),
            "Summary": metadata_payload.get("summary"),
        }
        for field_name, expected_value in expected_metadata.items():
            if not expected_value:
                continue
            actual_value = metadata_file_values[field_name]
            if actual_value != expected_value:
                issues.append(
                    f"{metadata_path}: {field_name} for {relative_path} must match owner-local metadata "
                    f"(metadata={actual_value!r}, doc={expected_value!r})."
                )

    for relative_path, entry in sorted(catalog_entries_by_path.items()):
        if relative_path not in expected_catalog_paths:
            issues.append(
                f"{catalog_path}: unexpected catalog procedure '{entry.registry_id}' for {relative_path}; "
                "add a matching cross-tool metadata contract or remove the registry metadata sidecar."
            )

    glossary_text = _extract_markdown_section(catalog_text, heading=RUNBOOK_STATUS_GLOSSARY_HEADING)
    if glossary_text is None:
        issues.append(f"{catalog_path}: missing '{RUNBOOK_STATUS_GLOSSARY_HEADING}' section.")
        return issues

    glossary_status_kinds = {
        match.group("kind").strip()
        for line in glossary_text.splitlines()
        for match in [PROGRESS_SURFACE_GLOSSARY_ROW_PATTERN.match(line.strip())]
        if match is not None
    }
    if not glossary_status_kinds:
        issues.append(f"{catalog_path}: status surface glossary section has no data table.")
        return issues

    for status_kind in sorted(expected_glossary_status_kinds - glossary_status_kinds):
        issues.append(f"{catalog_path}: missing status surface glossary entry for '{status_kind}'.")
    for status_kind in sorted(glossary_status_kinds - expected_glossary_status_kinds):
        issues.append(f"{catalog_path}: unexpected status surface glossary entry for '{status_kind}'.")

    return issues


def _extract_markdown_section(text: str, *, heading: str) -> str | None:
    lines = text.splitlines()
    try:
        heading_index = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return None

    section_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("### "):
            break
        section_lines.append(line)
    return "\n".join(section_lines)


def _find_study_record_doc_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    resolved_repo_root = repo_root.resolve()

    legacy_study_root = repo_root / "docs" / "studies" / "promoter"
    if legacy_study_root.exists():
        issues.append(f"{legacy_study_root}: legacy family-nested study record path must not exist.")

    for relative_path in STUDY_RECORD_REQUIRED_READMES:
        path = repo_root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if LEGACY_STUDY_RECORD_PREFIX in text:
            issues.append(
                f"{path}: legacy family-nested study path "
                f"'{LEGACY_STUDY_RECORD_PREFIX}<study-id>/...' must not appear; "
                "use 'docs/studies/<study-id>/...'."
            )
        documented_references = _collect_markdown_reference_names(text)
        for required_name in STUDY_RECORD_REQUIRED_FILES:
            if required_name not in documented_references:
                issues.append(
                    f"{path}: missing navigable study-record contract reference for '{required_name}' "
                    "as a markdown link or code span."
                )

    for relative_path in STUDY_RECORD_ROUTER_FILES:
        path = repo_root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if LEGACY_STUDY_INDEX_PATH in text:
            issues.append(
                f"{path}: legacy study index path '{LEGACY_STUDY_INDEX_PATH}' must not appear; "
                f"use '{ACTIVE_STUDY_INDEX_PATH}'."
            )
        if LEGACY_STUDY_RECORD_PREFIX in text:
            issues.append(
                f"{path}: legacy family-nested study path "
                f"'{LEGACY_STUDY_RECORD_PREFIX}<study-id>/...' must not appear; "
                "use 'docs/studies/<study-id>/...'."
            )
        if ACTIVE_STUDY_INDEX_PATH not in text:
            issues.append(f"{path}: study-record router must reference '{ACTIVE_STUDY_INDEX_PATH}'.")

    index_path = repo_root / "docs" / "studies" / "index.yaml"
    if not index_path.exists():
        return issues

    study_records_root = (repo_root / "docs" / "studies").resolve()
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return [f"{index_path}: study index must be a mapping."]

    version = payload.get("version")
    if version != 1:
        issues.append(f"{index_path}: study index must declare version: 1.")

    active_study = payload.get("active_study_id")
    studies_payload = payload.get("studies") or []
    if not isinstance(studies_payload, list):
        issues.append(f"{index_path}: 'studies' must be a list.")
        return issues

    entries_by_id: dict[str, Path] = {}
    for index, entry in enumerate(studies_payload, start=1):
        if not isinstance(entry, dict):
            issues.append(f"{index_path}: study entry {index} must be a mapping.")
            continue
        study_id = str(entry.get("study_id") or "").strip()
        raw_path = str(entry.get("record_root") or "").strip()
        if not study_id:
            issues.append(f"{index_path}: study entry {index} must define study_id.")
            continue
        if "family" in entry:
            issues.append(
                f"{index_path}: study entry {study_id!r} must not define legacy family; "
                "use the study record's explicit ops_surfaces instead."
            )
            continue
        if not raw_path:
            issues.append(f"{index_path}: study entry {study_id!r} must define record_root.")
            continue
        resolved_path = (
            (repo_root / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        )
        try:
            resolved_path.relative_to(resolved_repo_root)
        except ValueError:
            issues.append(f"{index_path}: study entry {study_id!r} path escapes the repository: {raw_path}")
            continue
        try:
            resolved_path.relative_to(study_records_root)
        except ValueError:
            issues.append(
                f"{index_path}: study entry {study_id!r} record_root must live under "
                f"docs/studies/<study-id> (path={raw_path!r})."
            )
            continue
        if resolved_path.name != study_id:
            issues.append(
                f"{index_path}: study entry {study_id!r} path must end with the same study id (path={raw_path!r})."
            )
        if study_id in entries_by_id:
            issues.append(f"{index_path}: duplicate study_id {study_id!r}.")
            continue
        entries_by_id[study_id] = resolved_path

    for study_id, study_root in sorted(entries_by_id.items()):
        issues.extend(_find_study_readme_frontmatter_issues(study_root=study_root, study_id=study_id))
        issues.extend(_find_study_ops_contract_layout_issues(study_root=study_root, study_id=study_id))

    active_study_text = str(active_study or "").strip() or None
    if active_study_text is None:
        issues.append(f"{index_path}: active_study_id must be a non-empty study id.")
        return issues
    if active_study_text not in entries_by_id:
        issues.append(f"{index_path}: active_study_id {active_study_text!r} is not declared under studies.")
        return issues

    study_root = entries_by_id[active_study_text]
    if not study_root.is_dir():
        issues.append(f"{index_path}: active study path is not a directory: {study_root}")
        return issues
    for required_name in STUDY_RECORD_REQUIRED_FILES:
        required_path = study_root / required_name
        if not required_path.exists():
            issues.append(
                f"{index_path}: active study {active_study_text!r} is missing "
                f"required file {required_name}: {required_path}"
            )

    return issues


def _find_study_ops_contract_layout_issues(*, study_root: Path, study_id: str) -> list[str]:
    issues: list[str] = []
    operations_root = study_root / "operations"
    ops_path = operations_root / "ops.study.yaml"
    if not ops_path.exists():
        return issues

    try:
        payload = yaml.safe_load(ops_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [f"{ops_path}: unable to parse ops.study.yaml ({exc})."]
    if not isinstance(payload, dict):
        return [f"{ops_path}: ops.study.yaml for study {study_id!r} must be a mapping."]

    legacy_pipeline_paths = (
        operations_root / "pipeline.yaml",
        operations_root / "runtime" / "pipeline.yaml",
    )
    for legacy_pipeline_path in legacy_pipeline_paths:
        if not legacy_pipeline_path.exists():
            continue
        issues.append(
            f"{legacy_pipeline_path}: study pipeline belongs under operations/runtime/command-groups/pipeline.yaml "
            "so OPS contracts, runtime plans, and contract fragments stay separate."
        )

    record_sources = payload.get("record_sources") or {}
    if record_sources and not isinstance(record_sources, dict):
        issues.append(f"{ops_path}: record_sources must be a mapping.")
    elif isinstance(record_sources, dict):
        pipeline_ref = str(record_sources.get("pipeline_ref") or "").strip()
        if pipeline_ref in STUDY_LEGACY_PIPELINE_REFS:
            issues.append(
                f"{ops_path}: record_sources.pipeline_ref must use {STUDY_RUNTIME_PIPELINE_REF!r}, "
                "not a legacy flat pipeline path."
            )

    parts = payload.get("parts")
    if parts is None:
        return issues
    if not isinstance(parts, dict):
        return [*issues, f"{ops_path}: parts must be a mapping."]

    unknown_parts = sorted(str(key) for key in parts if str(key) not in STUDY_OPS_CONTRACT_PART_KEYS)
    if unknown_parts:
        issues.append(f"{ops_path}: parts contains unknown section(s): {', '.join(unknown_parts)}.")

    for raw_section, raw_ref in parts.items():
        section = str(raw_section or "").strip()
        if section in payload:
            issues.append(f"{ops_path}: parts.{section} duplicates an inline {section} section.")
        if isinstance(raw_ref, list):
            raw_refs = raw_ref
            if not raw_refs:
                issues.append(f"{ops_path}: parts.{section} must list at least one operations-relative path.")
                continue
        else:
            raw_refs = [raw_ref]
        for index, raw_part_ref in enumerate(raw_refs):
            ref_label = f"parts.{section}" if len(raw_refs) == 1 else f"parts.{section}[{index}]"
            ref = str(raw_part_ref or "").strip()
            if not ref:
                issues.append(f"{ops_path}: {ref_label} must be a non-empty operations-relative path.")
                continue
            if ref.startswith(("repo:", "manifest:")):
                issues.append(f"{ops_path}: {ref_label} must be operations-relative, not a path ref.")
                continue
            part_rel = Path(ref)
            if part_rel.is_absolute() or ".." in part_rel.parts:
                issues.append(f"{ops_path}: {ref_label} must stay inside the operations directory.")
                continue
            if not part_rel.parts or part_rel.parts[0] != STUDY_OPS_CONTRACT_PARTS_DIR:
                issues.append(
                    f"{ops_path}: {ref_label} must live under operations/{STUDY_OPS_CONTRACT_PARTS_DIR}/ "
                    "to keep the root OPS record as a one-hop index."
                )
            part_path = operations_root / part_rel
            if not part_path.exists():
                issues.append(f"{ops_path}: {ref_label} references missing file {part_path}.")
                continue
            if part_path.suffix in {".yaml", ".yml"}:
                line_count = len(part_path.read_text(encoding="utf-8").splitlines())
                if line_count > STUDY_OPS_CONTRACT_PART_MAX_LINES:
                    issues.append(
                        f"{part_path}: ops contract part has {line_count} lines; split bulky owner lanes into "
                        f"semantic fragments below operations/{STUDY_OPS_CONTRACT_PARTS_DIR}/."
                    )
            continue

    return issues


def _find_study_readme_frontmatter_issues(*, study_root: Path, study_id: str) -> list[str]:
    issues: list[str] = []
    for relative_name, expected_surface, extra_key in (
        ("README.md", "study-root", "first_hop"),
        ("routes/README.md", "study-route-map", "entrypoint"),
    ):
        path = study_root / relative_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            issues.append(f"{path}: study navigation docs must start with YAML frontmatter.")
            continue
        try:
            raw_frontmatter = text.split("---", 2)[1]
            payload = yaml.safe_load(raw_frontmatter)
        except (IndexError, yaml.YAMLError) as exc:
            issues.append(f"{path}: unable to parse YAML frontmatter ({exc}).")
            continue
        if not isinstance(payload, dict):
            issues.append(f"{path}: YAML frontmatter must be a mapping.")
            continue
        missing = sorted(
            key
            for key in (*STUDY_README_FRONTMATTER_REQUIRED_KEYS, extra_key)
            if not str(payload.get(key) or "").strip()
        )
        if missing:
            issues.append(f"{path}: missing study navigation frontmatter key(s): {', '.join(missing)}.")
        if str(payload.get("study_id") or "").strip() != study_id:
            issues.append(f"{path}: frontmatter study_id must be {study_id!r}.")
        if str(payload.get("surface") or "").strip() != expected_surface:
            issues.append(f"{path}: frontmatter surface must be {expected_surface!r}.")
    return issues


def _find_study_status_surface_semantics_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    target_files = _collect_markdown_files_from_relative_paths(
        repo_root,
        relative_paths=STUDY_STATUS_SURFACE_SEMANTICS_DOC_PATHS,
    )
    for path in target_files:
        content = _markdown_text_without_fenced_code(path.read_text(encoding="utf-8"))
        for term in LEGACY_STUDY_STATUS_SURFACE_TERMS:
            if term not in content:
                continue
            line_no = content[: content.index(term)].count("\n") + 1
            issues.append(
                f"{path}:{line_no}: stale study-status ontology term {term!r} is not allowed; "
                "route studies through concrete study-owned providers only when those providers exist."
            )
    return issues


def _find_repo_local_skill_frontmatter_issues(repo_root: Path) -> list[str]:
    skills_root = repo_root / REPO_LOCAL_SKILLS_DIR
    if not skills_root.exists():
        return []

    issues: list[str] = []
    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            issues.append(f"{skill_file}: missing YAML frontmatter.")
            continue
        try:
            raw_frontmatter = text.split("---", 2)[1]
            payload = yaml.safe_load(raw_frontmatter)
        except (IndexError, yaml.YAMLError) as exc:
            issues.append(f"{skill_file}: unable to parse YAML frontmatter ({exc}).")
            continue
        if not isinstance(payload, dict):
            issues.append(f"{skill_file}: YAML frontmatter must be a mapping.")
            continue

        description = payload.get("description")
        if not isinstance(description, str) or not description.strip():
            issues.append(f"{skill_file}: frontmatter description must be a non-empty string.")
            continue
        description_length = len(description)
        if description_length > REPO_LOCAL_SKILL_DESCRIPTION_MAX_CHARS:
            issues.append(
                f"{skill_file}: frontmatter description length {description_length}/"
                f"{REPO_LOCAL_SKILL_DESCRIPTION_MAX_CHARS}; keep repo-local skill discovery compact."
            )

    return issues


def _collect_markdown_reference_names(text: str) -> set[str]:
    references: set[str] = set()
    for code_span in re.findall(r"`([^`\n]+)`", text):
        normalized = code_span.strip()
        if normalized:
            references.add(normalized)
            references.add(Path(normalized).name)
    for raw in LINK_PATTERN.findall(text):
        link = raw.strip().split()[0]
        target_rel = link.split("#", 1)[0].strip()
        if target_rel:
            references.add(target_rel)
            references.add(Path(target_rel).name)
    return references


def _find_active_shared_usr_dataset_id_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    resolved_repo_root = repo_root.resolve()
    index_path = repo_root / ACTIVE_STUDY_INDEX_PATH
    if not index_path.exists():
        return issues

    payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return issues

    active_study = str(payload.get("active_study_id") or "").strip()
    studies_payload = payload.get("studies") or []
    if not active_study or not isinstance(studies_payload, list):
        return issues

    study_root: Path | None = None
    for entry in studies_payload:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("study_id") or "").strip() != active_study:
            continue
        raw_path = str(entry.get("record_root") or "").strip()
        if not raw_path:
            return issues
        candidate = (repo_root / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        try:
            candidate.relative_to(resolved_repo_root)
        except ValueError:
            return issues
        study_root = candidate
        break

    if study_root is None:
        return issues

    datasets_path = study_root / "record" / "datasets.yaml"
    if not datasets_path.exists():
        return issues

    datasets_payload = yaml.safe_load(datasets_path.read_text(encoding="utf-8")) or {}
    if not isinstance(datasets_payload, dict):
        return issues
    entries = datasets_payload.get("datasets") or []
    if not isinstance(entries, list):
        return issues

    def is_disallowed_nested_id(dataset_id: str, *, status: str) -> bool:
        if "/" not in dataset_id:
            return False
        top_level = dataset_id.split("/", maxsplit=1)[0]
        return not (top_level == "archived" and status == "archived")

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        root_kind = str(entry.get("root_kind") or "").strip()
        if root_kind != "shared":
            continue

        role = str(entry.get("role") or index).strip()
        status = str(entry.get("status") or "").strip()
        dataset_id = str(entry.get("dataset") or "").strip()
        if dataset_id and is_disallowed_nested_id(dataset_id, status=status):
            issues.append(
                f"{datasets_path}: dataset entry {role!r} uses nested active shared "
                f"dataset id {dataset_id!r}. {ACTIVE_SHARED_USR_DATASET_ID_NUDGE}"
            )

        sync = entry.get("sync")
        if not isinstance(sync, dict):
            continue
        remote_root_kind = str(sync.get("remote_root_kind") or "").strip()
        remote_dataset = str(sync.get("remote_dataset") or "").strip()
        if remote_root_kind == "shared" and remote_dataset not in {"", "n/a"}:
            if is_disallowed_nested_id(remote_dataset, status=status):
                issues.append(
                    f"{datasets_path}: dataset entry {role!r} uses nested active shared "
                    f"remote_dataset id {remote_dataset!r}. {ACTIVE_SHARED_USR_DATASET_ID_NUDGE}"
                )

    return issues


def _find_shared_usr_dataset_layout_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    datasets_root = repo_root / SHARED_USR_DATASETS_ROOT
    if not datasets_root.exists():
        return issues

    for records_path in sorted(datasets_root.rglob("records.parquet")):
        dataset_dir = records_path.parent
        relative_dataset = dataset_dir.relative_to(datasets_root).as_posix()
        parts = Path(relative_dataset).parts
        if not parts or parts[0] == "archived":
            continue
        if len(parts) > 1:
            issues.append(
                f"{dataset_dir}: nested shared USR dataset root {relative_dataset!r} is not allowed. "
                f"{SHARED_USR_DATASET_LAYOUT_NUDGE}"
            )

    return issues


def _find_exec_plan_metadata_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    exec_root = repo_root / "docs" / "exec-plans"
    if not exec_root.exists():
        return issues

    for lane_name in ("active", "completed"):
        lane_root = exec_root / lane_name
        if not lane_root.exists():
            continue
        for plan_path in sorted(lane_root.rglob("*.md")):
            if plan_path.name == "README.md":
                continue

            text = plan_path.read_text(encoding="utf-8")
            present_sections = {heading.strip() for heading in SECTION_HEADING_PATTERN.findall(text)}
            section_bodies = _extract_section_bodies(text)

            status = _extract_metadata_field(text, STATUS_PATTERN)
            if status is None:
                issues.append(f"{plan_path}: missing '**Status:**' metadata field.")
                continue
            if status not in _EXEC_PLAN_STATUSES:
                allowed_statuses = ", ".join(sorted(_EXEC_PLAN_STATUSES))
                issues.append(f"{plan_path}: invalid status '{status}' (expected one of: {allowed_statuses}).")
                continue
            if lane_name == "completed" and status != "completed":
                issues.append(f"{plan_path}: plans under completed/ must set status to 'completed'.")
            if lane_name == "active" and status == "completed":
                issues.append(f"{plan_path}: plans under active/ cannot set status to 'completed'.")

            owner = _extract_metadata_field(text, OWNER_PATTERN)
            if owner is None:
                issues.append(f"{plan_path}: missing '**Owner:**' metadata field.")
            elif not owner:
                issues.append(f"{plan_path}: '**Owner:**' must not be empty.")

            created_raw = _extract_metadata_field(text, CREATED_PATTERN)
            if created_raw is None:
                issues.append(f"{plan_path}: missing '**Created:**' metadata field.")
            elif not created_raw:
                issues.append(f"{plan_path}: '**Created:**' must not be empty.")
            else:
                try:
                    _parse_iso_date(created_raw, field_name="Created", path=plan_path)
                except ValueError as exc:
                    issues.append(str(exc))

            if not LINK_PATTERN.search(text):
                issues.append(f"{plan_path}: execution plans must include at least one markdown link for traceability.")

            missing_sections = [name for name in _EXEC_PLAN_REQUIRED_SECTIONS if name not in present_sections]
            if missing_sections:
                missing_csv = ", ".join(missing_sections)
                issues.append(f"{plan_path}: missing required execution-plan sections: {missing_csv}.")
                continue

            progress_body = section_bodies.get("Progress", "")
            if not CHECKLIST_ITEM_PATTERN.search(progress_body):
                issues.append(f"{plan_path}: '## Progress' must include checklist items (e.g., '- [ ] ...').")
            progress_items = [line for line in progress_body.splitlines() if CHECKLIST_ITEM_PATTERN.match(line)]
            for progress_item in progress_items:
                if PROGRESS_ITEM_TIMESTAMP_PATTERN.match(progress_item):
                    continue
                issues.append(
                    f"{plan_path}: progress checklist items must include a UTC timestamp "
                    f"in '(YYYY-MM-DD HH:MMZ)' format."
                )
                break

            for section_name, body in section_bodies.items():
                if section_name == "Progress":
                    continue
                if CHECKLIST_ITEM_PATTERN.search(body):
                    issues.append(
                        f"{plan_path}: checklist items are only allowed under "
                        f"'## Progress' (found in '## {section_name}')."
                    )

    return issues


def _extract_section_bodies(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = SECTION_HEADING_PATTERN.match(line)
        if match is not None:
            current = match.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def _collect_public_interface_markdown_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set(_collect_tool_readme_markdown_files(repo_root))
    for rel in PUBLIC_INTERFACE_DOC_PATHS:
        target = repo_root / rel
        if not target.exists():
            continue
        if target.is_file() and target.suffix == ".md":
            files.add(target)
            continue
        if target.is_dir():
            for path in target.rglob("*.md"):
                files.add(path)
    return sorted(files)


def _find_public_interface_doc_contract_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for path in _collect_public_interface_markdown_files(repo_root):
        text = path.read_text(encoding="utf-8")
        for token in ABSOLUTE_DOC_PATH_TOKENS:
            if token in text:
                issues.append(
                    f"{path}: absolute filesystem path token '{token}' is not allowed; "
                    "use workspace-relative commands/paths."
                )
                break
        if INTERNAL_SOURCE_INREACH_PATTERN.search(text):
            issues.append(
                f"{path}: internal source inreach detected ('dnadesign.<tool>.src.*' or "
                "'src/dnadesign/<tool>/src/'); use public CLI/artifact contracts."
            )
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

        text = readme_path.read_text(encoding="utf-8")
        top_block = "\n".join(text.splitlines()[:25])
        banner_match = TOOL_README_BANNER_PATTERN.search(top_block)
        if banner_match is None:
            issues.append(f"{readme_path}: missing top banner image markdown line with '* banner' alt text.")
            continue

        link = banner_match.group("link").strip().split()[0]
        parsed = urlparse(link)
        if parsed.scheme or link.startswith("mailto:") or not link.lower().endswith(".svg"):
            issues.append(f"{readme_path}: banner link must target a local .svg asset.")
            continue

        target_rel = link.split("#", 1)[0].strip()
        if not target_rel:
            issues.append(f"{readme_path}: banner link must include a relative asset path.")
            continue

        target_path = (readme_path.parent / target_rel).resolve()
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

    return issues


def _find_tool_readme_structure_issues(repo_root: Path) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    issues: list[str] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        readme_path = src_root / tool_name / "README.md"
        if not readme_path.exists():
            continue

        text = readme_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) > TOOL_README_MAX_LINES:
            issues.append(
                f"{readme_path}: top-level tool README has {len(lines)} lines; "
                f"keep it at or below {TOOL_README_MAX_LINES} lines and route detail into docs/."
            )

        non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
        if not non_empty_indices:
            issues.append(f"{readme_path}: README is empty.")
            continue

        first_index = non_empty_indices[0]
        first_line = lines[first_index].strip()
        if TOOL_README_BANNER_PATTERN.search(first_line) is None:
            issues.append(f"{readme_path}: first non-empty line must be the banner image line.")
            continue

        next_index = next((idx for idx in non_empty_indices if idx > first_index), None)
        if next_index is not None and lines[next_index].lstrip().startswith("#"):
            issues.append(
                f"{readme_path}: line after the banner must be narrative text; avoid repeating a top title heading."
            )

        first_heading_index = next(
            (idx for idx in range(first_index + 1, len(lines)) if lines[idx].lstrip().startswith("#")),
            len(lines),
        )
        intro_lines = lines[first_index + 1 : first_heading_index]
        while intro_lines and not intro_lines[0].strip():
            intro_lines.pop(0)
        while intro_lines and not intro_lines[-1].strip():
            intro_lines.pop()

        if not intro_lines:
            issues.append(f"{readme_path}: banner must be followed by one narrative paragraph before docs links.")
        else:
            if any(not line.strip() for line in intro_lines):
                issues.append(
                    f"{readme_path}: intro after the banner must be one paragraph; route extra setup into docs/."
                )
            intro_text = " ".join(line.strip() for line in intro_lines)
            if TOOL_README_SELF_REFERENTIAL_INTRO_PATTERN.search(intro_text):
                issues.append(
                    f"{readme_path}: intro must describe what the tool does; avoid self-referential "
                    "package/layer-in-dnadesign wording."
                )

        if first_heading_index == len(lines):
            issues.append(f"{readme_path}: top-level tool README must include a '## Documentation' section.")
        elif lines[first_heading_index].strip() != "## Documentation":
            issues.append(f"{readme_path}: first heading after the intro must be '## Documentation' for the link map.")

        top_block = "\n".join(lines[:TOOL_README_TOP_LINK_SCAN_LINES])
        first_local_markdown_link: str | None = None
        for raw in LINK_PATTERN.findall(top_block):
            link = raw.strip().split()[0]
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_rel = link.split("#", 1)[0].strip()
            if target_rel.lower().endswith(".md"):
                first_local_markdown_link = target_rel
                break
        if first_local_markdown_link is None:
            issues.append(f"{readme_path}: top section must include a local markdown link to deeper documentation.")
            continue

        docs_index = None
        for candidate in (readme_path.parent / "docs" / "README.md", readme_path.parent / "docs" / "index.md"):
            if candidate.exists():
                docs_index = candidate.resolve()
                break
        if docs_index is None:
            continue

        first_target = (readme_path.parent / first_local_markdown_link).resolve()
        if first_target != docs_index:
            expected_rel = docs_index.relative_to(readme_path.parent).as_posix()
            issues.append(
                f"{readme_path}: first local markdown link must point to the tool docs index "
                f"'{expected_rel}', not '{first_local_markdown_link}'."
            )

    return issues


def _is_runbook_demo_doc(*, path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    if "/archived/" in rel or "/prototypes/" in rel:
        return False
    if rel.endswith("/runbook.md"):
        return True
    if "/docs/demos/" in rel:
        return True
    if "/docs/tutorials/" in rel:
        return True
    if "/docs/workflows/" in rel:
        return True
    if "/docs/howto/" in rel:
        return True
    if "/docs/operations/" in rel:
        return True
    if "/campaigns/demo_" in rel and rel.endswith("/README.md"):
        return True
    if rel.endswith("/workspaces/README.md"):
        return True
    if rel.startswith("src/dnadesign/densegen/workspaces/") and rel.endswith("/README.md"):
        return True
    return False


def _collect_runbook_demo_markdown_files(repo_root: Path) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    files: list[Path] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        tool_root = src_root / tool_name
        if not tool_root.exists():
            continue
        for path in sorted(tool_root.rglob("*.md")):
            if _is_runbook_demo_doc(path=path, repo_root=repo_root):
                files.append(path)
    return files


def _is_shell_control_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped in {"{", "}", ";;", "in", "PY"}:
        return True
    if stripped.endswith(" then"):
        return True
    if stripped.endswith("{"):
        return True
    if any(stripped.startswith(prefix) for prefix in RUNBOOK_DEMO_CONTROL_PREFIXES):
        return True
    if stripped.startswith(("cruncher() {", "dense() {")):
        return True
    return False


def _find_runbook_demo_snippet_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []

    for path in _collect_runbook_demo_markdown_files(repo_root):
        lines = path.read_text(encoding="utf-8").splitlines()
        line_idx = 0
        while line_idx < len(lines):
            line = lines[line_idx]
            if not line.startswith("```"):
                line_idx += 1
                continue

            lang = line[3:].strip().lower()
            if lang not in RUNBOOK_DEMO_SHELL_LANGS and lang not in RUNBOOK_DEMO_YAML_LANGS:
                line_idx += 1
                continue

            block_start = line_idx + 1  # 1-based line number of the first block line.
            block_lines: list[str] = []
            line_idx += 1
            while line_idx < len(lines) and not lines[line_idx].startswith("```"):
                block_lines.append(lines[line_idx])
                line_idx += 1

            if lang in RUNBOOK_DEMO_SHELL_LANGS:
                heredoc_end: str | None = None
                for idx, raw in enumerate(block_lines):
                    stripped = raw.strip()

                    if heredoc_end is not None:
                        if stripped == heredoc_end:
                            heredoc_end = None
                        continue

                    if _is_shell_control_line(raw):
                        continue

                    prev_non_empty: str | None = None
                    for prev in range(idx - 1, -1, -1):
                        previous = block_lines[prev].strip()
                        if previous:
                            prev_non_empty = block_lines[prev]
                            break

                    if prev_non_empty is not None and prev_non_empty.rstrip().endswith("\\"):
                        continue

                    has_inline_comment = " #" in raw
                    prev_is_comment = prev_non_empty is not None and prev_non_empty.strip().startswith("#")
                    if not has_inline_comment and not prev_is_comment:
                        line_no = block_start + idx
                        issues.append(f"{path}:{line_no}: command in shell block needs an explanatory comment.")

                    heredoc_match = RUNBOOK_DEMO_HEREDOC_PATTERN.search(stripped)
                    if heredoc_match is not None:
                        heredoc_end = heredoc_match.group(1)

            if lang in RUNBOOK_DEMO_YAML_LANGS:
                for idx, raw in enumerate(block_lines):
                    stripped = raw.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if ":" not in raw:
                        continue
                    if not RUNBOOK_DEMO_YAML_VALUE_PATTERN.match(raw):
                        continue
                    _, value = raw.split(":", 1)
                    value_text = value.strip()
                    if not value_text or value_text in {"|", ">"}:
                        continue
                    if "#" in value:
                        continue
                    line_no = block_start + idx
                    issues.append(
                        f"{path}:{line_no}: yaml key/value in runbook/demo snippets needs a right-side inline comment."
                    )

    return issues


def _is_ops_operational_runbook_contract(path: Path) -> bool:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: operational runbook yaml is invalid: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{path}: operational runbook yaml is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        return False
    runbook = payload.get("runbook")
    if not isinstance(runbook, dict):
        return False
    workflow_id = runbook.get("workflow_id")
    if not isinstance(workflow_id, str):
        return False
    return workflow_id in OPS_OPERATIONAL_WORKFLOW_IDS


def _is_allowed_operational_runbook_path(*, relative_path: Path) -> bool:
    for prefix in OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES:
        if relative_path == prefix or prefix in relative_path.parents:
            return True
    parts = relative_path.parts
    if "outputs" in parts and "logs" in parts and "ops" in parts and "runbooks" in parts:
        return True
    return False


def _find_operational_runbook_path_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for path in _iter_operational_runbook_candidate_yaml_files(repo_root):
        if not _is_ops_operational_runbook_contract(path):
            continue
        relative_path = path.relative_to(repo_root)
        if _is_allowed_operational_runbook_path(relative_path=relative_path):
            continue
        issues.append(
            f"{path}: operational runbook path is outside allowed locations; "
            "use workspace outputs/logs/ops/runbooks/ or src/dnadesign/ops/runbooks/presets/."
        )
    return issues


def _iter_operational_runbook_candidate_yaml_files(repo_root: Path):
    tracked_paths = _list_git_tracked_yaml_files(repo_root)
    if tracked_paths is not None:
        yield from tracked_paths
        return
    yield from _iter_bounded_operational_runbook_yaml_files(repo_root)


def _list_git_tracked_yaml_files(repo_root: Path) -> tuple[Path, ...] | None:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", "*.yaml", "*.yml"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "").strip().lower()
        if "not a git repository" in stderr:
            return None
        raise ValueError(f"git ls-files failed while collecting tracked yaml candidates: {stderr or exc}") from exc

    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_line in completed.stdout.splitlines():
        relative = Path(str(raw_line).strip())
        if not relative.parts:
            continue
        candidate = repo_root / relative
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".yaml", ".yml"}:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(candidate)
    return tuple(candidates)


def _iter_bounded_operational_runbook_yaml_files(repo_root: Path):
    seen: set[Path] = set()
    for suffix in ("*.yaml", "*.yml"):
        for path in sorted(repo_root.glob(suffix)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path
    for relative_root in OPS_OPERATIONAL_RUNBOOK_FALLBACK_SCAN_ROOTS:
        target = repo_root / relative_root
        if not target.exists():
            continue
        for suffix in ("*.yaml", "*.yml"):
            for path in sorted(target.rglob(suffix)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path


def _should_descend_operational_runbook_dir(relative_parts: tuple[str, ...]) -> bool:
    if not relative_parts:
        return True
    name = relative_parts[-1]
    if name in OPERATIONAL_RUNBOOK_SCAN_PRUNE_DIRS:
        return False
    if "outputs" in relative_parts:
        outputs_idx = relative_parts.index("outputs")
        tail = relative_parts[outputs_idx + 1 :]
        if not tail:
            return True
        if tail[0] != "logs":
            return False
        if len(tail) == 1:
            return True
        if tail[1] != "ops":
            return False
        if len(tail) == 2:
            return True
        if tail[2] != "runbooks":
            return False
    return True


def _find_packaged_runbook_variant_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    preset_root = repo_root / PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR
    if not preset_root.exists():
        return issues
    for suffix in ("*.yaml", "*.yml"):
        for path in sorted(preset_root.rglob(suffix)):
            if not path.is_file():
                continue
            if not _is_ops_operational_runbook_contract(path):
                continue
            if PACKAGED_RUNBOOK_DURATION_SUFFIX_PATTERN.search(path.stem) is None:
                continue
            issues.append(
                f"{path}: duration-suffixed operational variants are not allowed in presets; "
                "use workspace outputs/logs/ops/runbooks/."
            )
    return issues


def _find_transient_operational_artifact_path_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for dir_name in DISALLOWED_REPO_ROOT_OUTPUT_DIR_NAMES:
        candidate = repo_root / dir_name
        if not candidate.exists():
            continue
        if not candidate.is_dir():
            continue
        if not any(candidate.iterdir()):
            continue
        issues.append(
            f"{candidate}: generated artifact directory is not allowed at repository root; "
            "use a tool or study workspace outputs/ root instead."
        )
    for dir_name in TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES:
        candidate = repo_root / dir_name
        if not candidate.exists():
            continue
        if not candidate.is_dir():
            continue
        if not any(candidate.iterdir()):
            continue
        issues.append(
            f"{candidate}: transient operational artifact directory is not allowed at repo root; "
            "use workspace-scoped outputs/logs/ops paths or /scratch for disposable working state."
        )
    return issues


def _find_shared_utils_path_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for relative_path in DISALLOWED_SHARED_UTILS_PATHS:
        candidate = repo_root / relative_path
        if not candidate.exists():
            continue
        issues.append(f"{candidate}: shared utils package is not allowed; keep utilities under src/dnadesign/<tool>/.")
    return issues


def _find_stale_overlay_guard_term_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    target_files = _collect_markdown_files_from_relative_paths(repo_root, relative_paths=OVERLAY_GUARD_DOC_PATHS)
    for path in target_files:
        content = path.read_text(encoding="utf-8")
        for term in STALE_OVERLAY_GUARD_TERMS:
            if term in content:
                issues.append(
                    f"{path}: stale overlay guard term '{term}' is not allowed; "
                    "use usr-overlay-guard and overlay_namespace."
                )
    return issues


def _find_ops_deprecated_semantics_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    target_files = _collect_markdown_files_from_relative_paths(
        repo_root,
        relative_paths=OPS_DEPRECATED_SEMANTICS_DOC_PATHS,
    )
    for path in target_files:
        content = path.read_text(encoding="utf-8")
        for term in OPS_DEPRECATED_SEMANTICS_TERMS:
            if term in content:
                issues.append(
                    f"{path}: deprecated ops semantics term '{term}' is not allowed; "
                    "use transport-neutral workflow ids and the presets surface only."
                )
    return issues


def _find_study_execution_source_drift_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    target_files = _collect_markdown_files_from_relative_paths(
        repo_root,
        relative_paths=STUDY_EXECUTION_SOURCE_DOC_PATHS,
    )
    for path in target_files:
        content = path.read_text(encoding="utf-8")
        match = PIPELINE_ONLY_EXECUTION_SOURCE_PATTERN.search(content)
        if match is None:
            continue
        line_no = content[: match.start()].count("\n") + 1
        issues.append(
            f"{path}:{line_no}: docs must not claim pipeline.yaml is the only execution-surface source; "
            "use ops.study.yaml for OPS-facing execution surfaces and pipeline.yaml only for "
            "supplemental runtime context."
        )
    return issues


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check docs markdown naming and local links.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--changed-files-file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited repository-relative change list. Verification dates must cover changes to "
            "enforced docs; local dirty Markdown files are detected automatically."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root

    try:
        docs_md_files, all_md_files = _collect_markdown_files(repo_root)
        changed_doc_dates = collect_changed_doc_dates(
            repo_root,
            changed_files_file=args.changed_files_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    bad_names = _find_bad_doc_names(docs_md_files)
    if bad_names:
        print("Docs naming check failed: use kebab-case markdown filenames.")
        for path in bad_names:
            print(f" - {path}")
        return 1

    sor_metadata_issues = _find_sor_metadata_issues(repo_root, changed_doc_dates=changed_doc_dates)
    if sor_metadata_issues:
        print("Root system-of-record metadata check failed:")
        for issue in sor_metadata_issues:
            print(f" - {issue}")
        return 1

    index_metadata_issues = _find_index_metadata_issues(repo_root, changed_doc_dates=changed_doc_dates)
    if index_metadata_issues:
        print("Docs index metadata check failed:")
        for issue in index_metadata_issues:
            print(f" - {issue}")
        return 1

    runbook_metadata_issues = _find_runbook_metadata_issues(repo_root, changed_doc_dates=changed_doc_dates)
    if runbook_metadata_issues:
        print("Docs runbook metadata check failed:")
        for issue in runbook_metadata_issues:
            print(f" - {issue}")
        return 1

    tool_docs_metadata_issues = _find_tool_docs_metadata_issues(repo_root, changed_doc_dates=changed_doc_dates)
    if tool_docs_metadata_issues:
        print("Tool docs metadata check failed:")
        for issue in tool_docs_metadata_issues:
            print(f" - {issue}")
        return 1

    cross_tool_doc_metadata_issues = _find_cross_tool_doc_metadata_issues(repo_root)
    if cross_tool_doc_metadata_issues:
        print("Cross-tool doc metadata check failed:")
        for issue in cross_tool_doc_metadata_issues:
            print(f" - {issue}")
        return 1

    runbook_catalog_issues = _find_runbook_catalog_issues(repo_root)
    if runbook_catalog_issues:
        print("Runbook catalog check failed:")
        for issue in runbook_catalog_issues:
            print(f" - {issue}")
        return 1

    study_record_doc_issues = _find_study_record_doc_issues(repo_root)
    if study_record_doc_issues:
        print("Study record docs check failed:")
        for issue in study_record_doc_issues:
            print(f" - {issue}")
        return 1

    study_status_surface_semantics_issues = _find_study_status_surface_semantics_issues(repo_root)
    if study_status_surface_semantics_issues:
        print("Study status surface semantics check failed:")
        for issue in study_status_surface_semantics_issues:
            print(f" - {issue}")
        return 1

    repo_local_skill_frontmatter_issues = _find_repo_local_skill_frontmatter_issues(repo_root)
    if repo_local_skill_frontmatter_issues:
        print("Repo-local skill frontmatter check failed:")
        for issue in repo_local_skill_frontmatter_issues:
            print(f" - {issue}")
        return 1

    active_shared_usr_dataset_id_issues = _find_active_shared_usr_dataset_id_issues(repo_root)
    if active_shared_usr_dataset_id_issues:
        print("Active shared USR dataset id check failed:")
        for issue in active_shared_usr_dataset_id_issues:
            print(f" - {issue}")
        return 1

    shared_usr_dataset_layout_issues = _find_shared_usr_dataset_layout_issues(repo_root)
    if shared_usr_dataset_layout_issues:
        print("Shared USR dataset layout check failed:")
        for issue in shared_usr_dataset_layout_issues:
            print(f" - {issue}")
        return 1

    interface_doc_issues = _find_public_interface_doc_contract_issues(repo_root)
    if interface_doc_issues:
        print("Public interface docs contract check failed:")
        for issue in interface_doc_issues:
            print(f" - {issue}")
        return 1

    densegen_disallowed_term_issues = _find_densegen_disallowed_term_issues(repo_root)
    if densegen_disallowed_term_issues:
        print("DenseGen docs language check failed:")
        for issue in densegen_disallowed_term_issues:
            print(f" - {issue}")
        return 1

    construct_legacy_operator_doc_issues = _find_construct_legacy_operator_doc_issues(repo_root)
    if construct_legacy_operator_doc_issues:
        print("Construct operator docs contract check failed:")
        for issue in construct_legacy_operator_doc_issues:
            print(f" - {issue}")
        return 1

    legacy_contract_surface_doc_issues = _find_legacy_contract_surface_doc_issues(repo_root)
    if legacy_contract_surface_doc_issues:
        print("Legacy contract surface docs check failed:")
        for issue in legacy_contract_surface_doc_issues:
            print(f" - {issue}")
        return 1

    runbook_demo_snippet_issues = _find_runbook_demo_snippet_issues(repo_root)
    if runbook_demo_snippet_issues:
        print("Runbook/demo snippet annotation check failed:")
        for issue in runbook_demo_snippet_issues:
            print(f" - {issue}")
        return 1

    operational_runbook_path_issues = _find_operational_runbook_path_issues(repo_root)
    if operational_runbook_path_issues:
        print("Operational runbook path check failed:")
        for issue in operational_runbook_path_issues:
            print(f" - {issue}")
        return 1

    packaged_runbook_variant_issues = _find_packaged_runbook_variant_issues(repo_root)
    if packaged_runbook_variant_issues:
        print("Packaged runbook variant check failed:")
        for issue in packaged_runbook_variant_issues:
            print(f" - {issue}")
        return 1

    transient_operational_artifact_path_issues = _find_transient_operational_artifact_path_issues(repo_root)
    if transient_operational_artifact_path_issues:
        print("Transient operational artifact placement check failed:")
        for issue in transient_operational_artifact_path_issues:
            print(f" - {issue}")
        return 1

    shared_utils_path_issues = _find_shared_utils_path_issues(repo_root)
    if shared_utils_path_issues:
        print("Shared utils path check failed:")
        for issue in shared_utils_path_issues:
            print(f" - {issue}")
        return 1

    stale_overlay_guard_term_issues = _find_stale_overlay_guard_term_issues(repo_root)
    if stale_overlay_guard_term_issues:
        print("Overlay guard terminology check failed:")
        for issue in stale_overlay_guard_term_issues:
            print(f" - {issue}")
        return 1

    ops_deprecated_semantics_issues = _find_ops_deprecated_semantics_issues(repo_root)
    if ops_deprecated_semantics_issues:
        print("Ops terminology drift check failed:")
        for issue in ops_deprecated_semantics_issues:
            print(f" - {issue}")
        return 1

    study_execution_source_drift_issues = _find_study_execution_source_drift_issues(repo_root)
    if study_execution_source_drift_issues:
        print("Study execution-source docs check failed:")
        for issue in study_execution_source_drift_issues:
            print(f" - {issue}")
        return 1

    tool_readme_banner_issues = _find_tool_readme_banner_issues(repo_root)
    if tool_readme_banner_issues:
        print("Tool README banner contract check failed:")
        for issue in tool_readme_banner_issues:
            print(f" - {issue}")
        return 1

    tool_readme_structure_issues = _find_tool_readme_structure_issues(repo_root)
    if tool_readme_structure_issues:
        print("Tool README structure contract check failed:")
        for issue in tool_readme_structure_issues:
            print(f" - {issue}")
        return 1

    readme_tool_catalog_issues = _find_readme_tool_catalog_issues(repo_root)
    if readme_tool_catalog_issues:
        print("README tool catalog check failed:")
        for issue in readme_tool_catalog_issues:
            print(f" - {issue}")
        return 1

    root_docs_entrypoint_issues = _find_root_docs_entrypoint_issues(repo_root)
    if root_docs_entrypoint_issues:
        print("Root docs entrypoint check failed:")
        for issue in root_docs_entrypoint_issues:
            print(f" - {issue}")
        return 1

    deprecated_docs_entrypoint_issues = _find_deprecated_docs_entrypoint_issues(repo_root)
    if deprecated_docs_entrypoint_issues:
        print("Deprecated docs entrypoint check failed:")
        for issue in deprecated_docs_entrypoint_issues:
            print(f" - {issue}")
        return 1

    docs_root_heading_style_issues = _find_docs_root_heading_style_issues(repo_root)
    if docs_root_heading_style_issues:
        print("Docs root heading style check failed:")
        for issue in docs_root_heading_style_issues:
            print(f" - {issue}")
        return 1

    entrypoint_local_path_issues = _find_entrypoint_local_path_literal_issues(repo_root)
    if entrypoint_local_path_issues:
        print("Entrypoint local path hyperlink check failed:")
        for issue in entrypoint_local_path_issues:
            print(f" - {issue}")
        return 1

    agents_path_reference_issues = _find_agents_path_reference_issues(repo_root)
    if agents_path_reference_issues:
        print("AGENTS path reference check failed:")
        for issue in agents_path_reference_issues:
            print(f" - {issue}")
        return 1

    codecov_component_issues = _find_codecov_component_issues(repo_root)
    if codecov_component_issues:
        print("Codecov component contract check failed:")
        for issue in codecov_component_issues:
            print(f" - {issue}")
        return 1

    exec_plan_issues = _find_exec_plan_metadata_issues(repo_root)
    if exec_plan_issues:
        print("Execution plan metadata check failed:")
        for issue in exec_plan_issues:
            print(f" - {issue}")
        return 1

    broken = _find_broken_links(all_md_files, repo_root=repo_root)
    if broken:
        print("Docs link check failed:")
        for src, link in broken:
            print(f" - {src}: {link}")
        return 1

    print(f"Docs checks passed ({len(all_md_files)} markdown files, including root system-of-record docs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
