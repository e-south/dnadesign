"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/check_contracts.py

Shared immutable policy values for documentation checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

from dnadesign.ops.runbooks import REPO_TRANSIENT_OPERATIONAL_DIR_NAMES

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
README_TOOL_LINK_PATTERN = re.compile(r"\[\*\*(?P<tool>[a-z0-9_-]+)\*\*\]\((?P<link>[^)]+)\)")
README_TOOL_COMPONENT_COVERAGE_PATTERN = re.compile(r"codecov\.io/[^\s)]+[?&]component=", flags=re.IGNORECASE)
TOOL_README_BANNER_LABEL_PATTERN = re.compile(r"\bbanner\b", flags=re.IGNORECASE)
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
    "src/dnadesign/cruncher/docs/demos",
    "src/dnadesign/cruncher/docs/reference/cli.md",
    "src/dnadesign/cruncher/workspaces",
    "src/dnadesign/densegen/README.md",
    "src/dnadesign/densegen/docs/howto",
    "src/dnadesign/densegen/docs/tutorials",
    "src/dnadesign/densegen/workspaces/README.md",
    "src/dnadesign/ops/README.md",
    "src/dnadesign/ops/docs",
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
    "src/dnadesign/baserender/docs/integrations/junction.md": {
        "type": "route",
        "plane": "downstream-tool",
        "owner_boundary": "baserender",
    },
}
_CROSS_TOOL_DOC_ALLOWED_TYPES = {"contract", "route", "runbook", "workflow"}
_CROSS_TOOL_DOC_ALLOWED_PLANES = {"control-plane", "data-plane", "downstream-tool"}
_RUNBOOK_CATALOG_METADATA_TYPES = {"contract", "runbook", "workflow"}
_REGISTRY_ID_VALUE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)+$")
_METADATA_TOKEN_VALUE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*(?:-[a-z0-9]+)*$")
RUNBOOK_CATALOG_DOC_PATH = "docs/runbooks/README.md"
RUNBOOK_STATUS_GLOSSARY_HEADING = "### Status views"
OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES = (Path("docs/templates"),)
OPS_OPERATIONAL_RUNBOOK_FALLBACK_SCAN_ROOTS = (Path("docs/templates"),)
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
    "src/dnadesign/ops/README.md",
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
