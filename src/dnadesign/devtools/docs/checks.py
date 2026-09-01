"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/checks.py

Coordinates documentation checks in their established command-line order.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnadesign.devtools.docs import banner_contracts as _banner_contracts
from dnadesign.devtools.docs.badges import find_markdown_badge_policy_issues
from dnadesign.devtools.docs.banner_contracts import (
    _find_banner_catalog_inventory_issues as _find_banner_catalog_inventory_issues,
)
from dnadesign.devtools.docs.banner_contracts import (
    _find_banner_source_drift_issues as _find_banner_source_drift_issues_impl,
)
from dnadesign.devtools.docs.banner_contracts import (
    _find_tool_readme_banner_issues as _find_tool_readme_banner_issues_impl,
)
from dnadesign.devtools.docs.banner_contracts import (
    _resolve_readme_banner_reference as _resolve_readme_banner_reference,
)
from dnadesign.devtools.docs.banner_contracts import (
    _top_rendered_readme_banner as _top_rendered_readme_banner,
)
from dnadesign.devtools.docs.banners.catalog import BANNERS
from dnadesign.devtools.docs.banners.render import check_banners
from dnadesign.devtools.docs.check_contracts import (
    OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES as OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES,
)
from dnadesign.devtools.docs.check_contracts import (
    TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES as TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES,
)
from dnadesign.devtools.docs.document_metadata import (
    _collect_markdown_reference_names as _collect_markdown_reference_names,
)
from dnadesign.devtools.docs.document_metadata import (
    _extract_doc_metadata_field as _extract_doc_metadata_field,
)
from dnadesign.devtools.docs.document_metadata import (
    _extract_markdown_section as _extract_markdown_section,
)
from dnadesign.devtools.docs.document_metadata import (
    _extract_metadata_field as _extract_metadata_field,
)
from dnadesign.devtools.docs.document_metadata import (
    _extract_section_bodies as _extract_section_bodies,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_cross_tool_doc_metadata_issues as _find_cross_tool_doc_metadata_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_exec_plan_metadata_issues as _find_exec_plan_metadata_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_index_metadata_issues as _find_index_metadata_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_owner_last_verified_metadata_issues as _find_owner_last_verified_metadata_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_owner_last_verified_metadata_issues_for_files as _find_owner_last_verified_metadata_issues_for_files,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_repo_local_skill_frontmatter_issues as _find_repo_local_skill_frontmatter_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_runbook_catalog_issues as _find_runbook_catalog_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_runbook_metadata_issues as _find_runbook_metadata_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_shared_usr_dataset_layout_issues as _find_shared_usr_dataset_layout_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_sor_metadata_issues as _find_sor_metadata_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _find_tool_docs_metadata_issues as _find_tool_docs_metadata_issues,
)
from dnadesign.devtools.docs.document_metadata import (
    _load_markdown_frontmatter as _load_markdown_frontmatter,
)
from dnadesign.devtools.docs.document_metadata import (
    _markdown_body_without_frontmatter as _markdown_body_without_frontmatter,
)
from dnadesign.devtools.docs.document_metadata import (
    _parse_iso_date as _parse_iso_date,
)
from dnadesign.devtools.docs.freshness import collect_changed_doc_dates
from dnadesign.devtools.docs.landing import (
    find_landing_readme_frontmatter_issues as _find_landing_readme_frontmatter_issues,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_markdown_anchors as _collect_markdown_anchors,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_markdown_files as _collect_markdown_files,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_markdown_files_from_relative_paths as _collect_markdown_files_from_relative_paths,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_markdown_headings_outside_fences as _collect_markdown_headings_outside_fences,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_tool_docs_markdown_files as _collect_tool_docs_markdown_files,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_tool_readme_markdown_files as _collect_tool_readme_markdown_files,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _extract_level2_section_lines as _extract_level2_section_lines,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _find_bad_doc_names as _find_bad_doc_names,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _find_broken_links as _find_broken_links,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _markdown_text_without_fenced_code as _markdown_text_without_fenced_code,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _normalize_relative_markdown_path as _normalize_relative_markdown_path,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _readme_tool_table_rows as _readme_tool_table_rows,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _slugify_markdown_heading as _slugify_markdown_heading,
)
from dnadesign.devtools.docs.operations_contracts import (
    _collect_runbook_demo_markdown_files as _collect_runbook_demo_markdown_files,
)
from dnadesign.devtools.docs.operations_contracts import (
    _find_operational_runbook_path_issues as _find_operational_runbook_path_issues,
)
from dnadesign.devtools.docs.operations_contracts import (
    _find_ops_deprecated_semantics_issues as _find_ops_deprecated_semantics_issues,
)
from dnadesign.devtools.docs.operations_contracts import (
    _find_runbook_demo_snippet_issues as _find_runbook_demo_snippet_issues,
)
from dnadesign.devtools.docs.operations_contracts import (
    _find_shared_utils_path_issues as _find_shared_utils_path_issues,
)
from dnadesign.devtools.docs.operations_contracts import (
    _find_stale_overlay_guard_term_issues as _find_stale_overlay_guard_term_issues,
)
from dnadesign.devtools.docs.operations_contracts import (
    _find_transient_operational_artifact_path_issues as _find_transient_operational_artifact_path_issues,
)
from dnadesign.devtools.docs.operations_contracts import (
    _is_allowed_operational_runbook_path as _is_allowed_operational_runbook_path,
)
from dnadesign.devtools.docs.operations_contracts import (
    _is_ops_operational_runbook_contract as _is_ops_operational_runbook_contract,
)
from dnadesign.devtools.docs.operations_contracts import (
    _is_runbook_demo_doc as _is_runbook_demo_doc,
)
from dnadesign.devtools.docs.operations_contracts import (
    _is_shell_control_line as _is_shell_control_line,
)
from dnadesign.devtools.docs.operations_contracts import (
    _iter_bounded_operational_runbook_yaml_files as _iter_bounded_operational_runbook_yaml_files,
)
from dnadesign.devtools.docs.operations_contracts import (
    _iter_operational_runbook_candidate_yaml_files as _iter_operational_runbook_candidate_yaml_files,
)
from dnadesign.devtools.docs.operations_contracts import (
    _list_git_tracked_yaml_files as _list_git_tracked_yaml_files,
)
from dnadesign.devtools.docs.operations_contracts import (
    _should_descend_operational_runbook_dir as _should_descend_operational_runbook_dir,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _collect_public_interface_markdown_files as _collect_public_interface_markdown_files,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_agents_path_reference_issues as _find_agents_path_reference_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_codecov_component_issues as _find_codecov_component_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_construct_legacy_operator_doc_issues as _find_construct_legacy_operator_doc_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_densegen_disallowed_term_issues as _find_densegen_disallowed_term_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_deprecated_docs_entrypoint_issues as _find_deprecated_docs_entrypoint_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_docs_root_heading_style_issues as _find_docs_root_heading_style_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_entrypoint_local_path_literal_issues as _find_entrypoint_local_path_literal_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_legacy_contract_surface_doc_issues as _find_legacy_contract_surface_doc_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_public_interface_doc_contract_issues as _find_public_interface_doc_contract_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_readme_tool_catalog_issues as _find_readme_tool_catalog_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_root_docs_entrypoint_issues as _find_root_docs_entrypoint_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _find_tool_readme_structure_issues as _find_tool_readme_structure_issues,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _resolve_agents_path_literal as _resolve_agents_path_literal,
)
from dnadesign.devtools.docs.public_surface_contracts import (
    _should_skip_agents_path_literal as _should_skip_agents_path_literal,
)


def _find_tool_readme_banner_issues(repo_root: Path) -> list[str]:
    """Compatibility seam for callers that patch the historical module globals."""
    _banner_contracts.BANNERS = BANNERS
    return _find_tool_readme_banner_issues_impl(repo_root)


def _find_banner_source_drift_issues(repo_root: Path) -> list[str]:
    """Compatibility seam for callers that patch the historical module globals."""
    _banner_contracts.check_banners = check_banners
    return _find_banner_source_drift_issues_impl(repo_root)


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

    repo_local_skill_frontmatter_issues = _find_repo_local_skill_frontmatter_issues(repo_root)
    if repo_local_skill_frontmatter_issues:
        print("Repo-local skill frontmatter check failed:")
        for issue in repo_local_skill_frontmatter_issues:
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

    banner_source_drift_issues = _find_banner_source_drift_issues(repo_root)
    if banner_source_drift_issues:
        print("Banner source drift check failed:")
        for issue in banner_source_drift_issues:
            print(f" - {issue}")
        return 1

    landing_readme_frontmatter_issues = _find_landing_readme_frontmatter_issues(repo_root)
    if landing_readme_frontmatter_issues:
        print("Landing README frontmatter check failed:")
        for issue in landing_readme_frontmatter_issues:
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

    markdown_badge_policy_issues = find_markdown_badge_policy_issues(repo_root, all_md_files)
    if markdown_badge_policy_issues:
        print("Markdown badge policy check failed:")
        for issue in markdown_badge_policy_issues:
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
