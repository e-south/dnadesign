"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_checks.py

Tests for docs naming/link validation checks used in CI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import pytest
import yaml

from dnadesign.devtools.docs.checks import (
    _find_active_shared_usr_dataset_id_issues,
    _find_agents_path_reference_issues,
    _find_broken_links,
    _find_cross_tool_doc_metadata_issues,
    _find_densegen_disallowed_term_issues,
    _find_deprecated_docs_entrypoint_issues,
    _find_docs_root_heading_style_issues,
    _find_entrypoint_local_path_literal_issues,
    _find_legacy_contract_surface_doc_issues,
    _find_operational_runbook_path_issues,
    _find_ops_deprecated_semantics_issues,
    _find_packaged_runbook_variant_issues,
    _find_public_interface_doc_contract_issues,
    _find_readme_tool_catalog_issues,
    _find_repo_local_skill_frontmatter_issues,
    _find_root_docs_entrypoint_issues,
    _find_runbook_catalog_issues,
    _find_runbook_demo_snippet_issues,
    _find_shared_usr_dataset_layout_issues,
    _find_shared_utils_path_issues,
    _find_stale_overlay_guard_term_issues,
    _find_study_execution_source_drift_issues,
    _find_study_record_doc_issues,
    _find_study_status_surface_semantics_issues,
    _find_tool_docs_metadata_issues,
    _find_tool_readme_banner_issues,
    _find_tool_readme_structure_issues,
    _find_transient_operational_artifact_path_issues,
    main,
)
from dnadesign.ops.catalog import (
    load_runbook_catalog,
    render_catalog_procedure_section,
    render_catalog_tool_source_section,
)
from dnadesign.ops.runbooks import (
    PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR,
    REPO_TRANSIENT_OPERATIONAL_DIR_NAMES,
)

VALID_TOOL_BANNER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="180" viewBox="0 0 1200 180"></svg>\n'
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_required_study_record_files(study_root: Path) -> None:
    for required_name in (
        "record/campaign.yaml",
        "record/datasets.yaml",
        "record/status.md",
        "operations/ops.study.yaml",
    ):
        _write(study_root / required_name, "placeholder\n")


def _git_init(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)


def _git_add(repo_root: Path, *paths: str) -> None:
    subprocess.run(["git", "add", *paths], cwd=repo_root, check=True, capture_output=True, text=True)


def _write_registry_metadata(
    doc_path: Path,
    *,
    catalog_order: int,
    registry_id: str,
    entry_type: str,
    plane: str,
    owner_boundary: str,
    entry_artifact: str,
    exit_artifact: str,
    summary: str,
    execution_kind: str,
    status_kind: str,
    relations: list[dict[str, str]] | None = None,
) -> None:
    metadata_path = doc_path.with_name(f"{doc_path.stem}.registry.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "catalog_order": catalog_order,
                "registry_id": registry_id,
                "type": entry_type,
                "plane": plane,
                "owner_boundary": owner_boundary,
                "entry_artifact": entry_artifact,
                "exit_artifact": exit_artifact,
                "summary": summary,
                "execution_kind": execution_kind,
                "status_kind": status_kind,
                "relations": relations or [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_tool_source_metadata(
    doc_path: Path,
    *,
    catalog_order: int,
    tool: str,
    summary: str,
    keywords: list[str] | None = None,
) -> None:
    metadata_path = doc_path.with_name(f"{doc_path.stem}.tool-source.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "catalog_order": catalog_order,
                "tool": tool,
                "summary": summary,
                "keywords": keywords or [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_runbook_catalog_readme(
    repo_root: Path,
    *,
    procedure_section: str,
    tool_source_section: str,
    glossary_rows: list[str],
) -> None:
    _write(
        repo_root / "docs" / "runbooks" / "README.md",
        "\n".join(
            [
                "## Runbook Catalog",
                "",
                "### Cross-tool procedures",
                "",
                procedure_section,
                "",
                "### Tool docs",
                "",
                tool_source_section,
                "",
                "### Status views",
                "",
                "| Status kind | Meaning | Check next |",
                "| --- | --- | --- |",
                *glossary_rows,
            ]
        )
        + "\n",
    )


def _write_generated_runbook_catalog_readme(repo_root: Path, *, glossary_rows: list[str]) -> None:
    _write_runbook_catalog_readme(
        repo_root,
        procedure_section="_placeholder_",
        tool_source_section="_placeholder_",
        glossary_rows=glossary_rows,
    )
    catalog = load_runbook_catalog(repo_root=repo_root)
    _write_runbook_catalog_readme(
        repo_root,
        procedure_section=render_catalog_procedure_section(catalog),
        tool_source_section=render_catalog_tool_source_section(catalog),
        glossary_rows=glossary_rows,
    )


def _empty_tool_source_section() -> str:
    return "\n".join(
        [
            (
                "This table is generated from owner-local `*.tool-source.yaml` metadata sidecars. "
                "Edit those files instead of hand-editing rows here."
            ),
            "",
            "| Tool | Docs entrypoint | What you will find |",
            "| --- | --- | --- |",
        ]
    )


def test_main_fails_when_docs_directory_is_missing(tmp_path: Path) -> None:
    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_for_non_kebab_docs_filename(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "bad_name.md", "# Bad\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_for_broken_relative_link(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "[missing](./nope.md)\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_broken_links_check_rejects_absolute_local_path_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    source = tmp_path / "docs" / "index.md"
    _write(source, f"[outside]({outside})\n")

    broken = _find_broken_links([source], repo_root=tmp_path)

    assert broken == [(source, f"{outside} (local link escapes repository)")]


def test_broken_link_check_ignores_fenced_code_markdown_links(tmp_path: Path) -> None:
    index_path = tmp_path / "docs" / "index.md"
    _write(
        index_path,
        "\n".join(
            [
                "## Examples",
                "",
                "```md",
                "[illustrative missing link](./not-a-real-route.md)",
                "```",
                "",
            ]
        ),
    )

    broken = _find_broken_links([index_path])

    assert broken == []


def test_broken_link_check_still_flags_body_markdown_links(tmp_path: Path) -> None:
    index_path = tmp_path / "docs" / "index.md"
    _write(index_path, "[missing](./nope.md)\n")

    broken = _find_broken_links([index_path])

    assert broken == [(index_path, "./nope.md")]


def test_find_study_record_doc_issues_flags_legacy_router_index_paths(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "studies" / "README.md",
        "\n".join(
            [
                "`record/campaign.yaml`",
                "`record/datasets.yaml`",
                "`record/status.md`",
                "`operations/ops.study.yaml`",
                "legacy path: docs/studies/promoter/demo_study/status.md",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "docs" / "studies" / "index.yaml",
        yaml.safe_dump(
            {
                "version": 1,
                "active_study_id": "demo_study",
                "studies": [
                    {
                        "study_id": "demo_study",
                        "record_root": "docs/studies/demo_study",
                    }
                ],
            },
            sort_keys=False,
        ),
    )
    _write_required_study_record_files(tmp_path / "docs" / "studies" / "demo_study")
    _write(tmp_path / "AGENTS.md", "- Promoter study active-study registry: `docs/studies/promoter/index.yaml`\n")
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "AGENTS.md",
        "- Active promoter-study registry: `docs/studies/promoter/index.yaml`\n",
    )

    issues = _find_study_record_doc_issues(tmp_path)

    assert any("AGENTS.md" in issue and "docs/studies/promoter/index.yaml" in issue for issue in issues)
    assert any(
        "src/dnadesign/usr/AGENTS.md" in issue and "docs/studies/promoter/index.yaml" in issue for issue in issues
    )
    assert any("AGENTS.md" in issue and "docs/studies/index.yaml" in issue for issue in issues)
    assert any("docs/studies/README.md" in issue and "docs/studies/promoter/" in issue for issue in issues)


def test_find_study_record_doc_issues_requires_navigable_required_file_references(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "studies" / "README.md",
        "Study records normally include campaign.yaml, datasets.yaml, status.md, and ops.study.yaml.\n",
    )

    issues = _find_study_record_doc_issues(tmp_path)

    assert any("campaign.yaml" in issue and "markdown link or code span" in issue for issue in issues)
    assert any("datasets.yaml" in issue and "markdown link or code span" in issue for issue in issues)
    assert any("status.md" in issue and "markdown link or code span" in issue for issue in issues)
    assert any("ops.study.yaml" in issue and "markdown link or code span" in issue for issue in issues)


def test_find_study_record_doc_issues_accepts_code_span_required_file_references(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "studies" / "README.md",
        "\n".join(
            [
                "- `record/campaign.yaml`",
                "- `record/datasets.yaml`",
                "- `record/status.md`",
                "- `operations/ops.study.yaml`",
            ]
        )
        + "\n",
    )

    issues = _find_study_record_doc_issues(tmp_path)

    assert not any("missing navigable study-record contract reference" in issue for issue in issues)


def test_find_study_record_doc_issues_rejects_record_root_outside_study_records(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "studies" / "README.md",
        "\n".join(
            [
                "- `record/campaign.yaml`",
                "- `record/datasets.yaml`",
                "- `record/status.md`",
                "- `operations/ops.study.yaml`",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "docs" / "studies" / "index.yaml",
        yaml.safe_dump(
            {
                "version": 1,
                "active_study_id": "demo_study",
                "studies": [
                    {
                        "study_id": "demo_study",
                        "record_root": "docs/other/demo_study",
                    }
                ],
            },
            sort_keys=False,
        ),
    )
    _write_required_study_record_files(tmp_path / "docs" / "other" / "demo_study")

    issues = _find_study_record_doc_issues(tmp_path)

    assert any("record_root must live under docs/studies/<study-id>" in issue for issue in issues)


def test_find_study_status_surface_semantics_issues_rejects_family_routing_terms(tmp_path: Path) -> None:
    _write(
        tmp_path / "ARCHITECTURE.md",
        "Study status adapters are explicit seams. "
        "Study-family adapters are explicit seams and family routing resolves through "
        "`src/dnadesign/studies/families/<family>/`.\n",
    )
    _write(
        tmp_path / "docs" / "README.md",
        "Verify the active study selector, `family`, and `record_root` in the study index.\n",
    )
    _write(
        tmp_path / "docs" / "studies" / "stress_ethanol_cipro_growth" / "contracts" / "status.md",
        "The selected study entry must declare `family` and `record_root`.\n",
    )

    issues = _find_study_status_surface_semantics_issues(tmp_path)

    assert any("Study status adapters" in issue for issue in issues)
    assert any("Study-family adapters" in issue for issue in issues)
    assert any("`family`, and `record_root`" in issue for issue in issues)
    assert any("declare `family` and `record_root`" in issue for issue in issues)


def test_find_repo_local_skill_frontmatter_issues_rejects_overlong_description(tmp_path: Path) -> None:
    _write(
        tmp_path / ".agents" / "skills" / "demo-skill" / "SKILL.md",
        "\n".join(
            [
                "---",
                "name: demo-skill",
                f"description: {'x' * 221}",
                "metadata:",
                "  version: 0.1.0",
                "  category: workflow-automation",
                "---",
                "",
                "# Demo Skill",
            ]
        )
        + "\n",
    )

    issues = _find_repo_local_skill_frontmatter_issues(tmp_path)

    assert any("frontmatter description length 221/220" in issue for issue in issues)


def _write_active_study_datasets(tmp_path: Path, datasets: list[dict[str, object]]) -> None:
    _write(
        tmp_path / "docs" / "studies" / "index.yaml",
        yaml.safe_dump(
            {
                "version": 1,
                "active_study_id": "demo_study",
                "studies": [
                    {
                        "study_id": "demo_study",
                        "record_root": "docs/studies/demo_study",
                    }
                ],
            },
            sort_keys=False,
        ),
    )
    _write(
        tmp_path / "docs" / "studies" / "demo_study" / "record" / "datasets.yaml",
        yaml.safe_dump({"datasets": datasets}, sort_keys=False),
    )


def test_active_shared_usr_dataset_id_check_flags_nested_shared_ids(tmp_path: Path) -> None:
    _write_active_study_datasets(
        tmp_path,
        [
            {
                "role": "densegen_anchor",
                "dataset": "densegen/demo_nested_source",
                "root_kind": "shared",
                "status": "present",
                "sync": {
                    "remote_dataset": "promoter/demo_anchor",
                    "remote_root_kind": "shared",
                },
            }
        ],
    )

    issues = _find_active_shared_usr_dataset_id_issues(tmp_path)

    assert any("densegen/demo_nested_source" in issue for issue in issues)
    assert any("remote_dataset" in issue and "promoter/demo_anchor" in issue for issue in issues)
    assert all("Active shared USR dataset IDs must be flat owner-first IDs" in issue for issue in issues)
    assert all("archived/ is the only special top-level bucket" in issue for issue in issues)


def test_active_shared_usr_dataset_id_check_allows_flat_ids_and_archived_bucket(tmp_path: Path) -> None:
    _write_active_study_datasets(
        tmp_path,
        [
            {
                "role": "densegen_anchor",
                "dataset": "densegen_prom_eth_cip_source",
                "root_kind": "shared",
                "status": "present",
                "sync": {
                    "remote_dataset": "densegen_prom_eth_cip_source",
                    "remote_root_kind": "shared",
                },
            },
            {
                "role": "archived_prior_anchor",
                "dataset": "archived/promoter/prior_anchor",
                "root_kind": "shared",
                "status": "archived",
                "sync": {
                    "remote_dataset": "archived/promoter/prior_anchor",
                    "remote_root_kind": "shared",
                },
            },
        ],
    )

    assert _find_active_shared_usr_dataset_id_issues(tmp_path) == []


def test_shared_usr_dataset_layout_check_flags_nested_dataset_roots(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "datasets" / "densegen" / "demo_sampling_baseline" / "records.parquet",
        "placeholder\n",
    )

    issues = _find_shared_usr_dataset_layout_issues(tmp_path)

    assert any("densegen/demo_sampling_baseline" in issue for issue in issues)
    assert all("Shared repo USR dataset roots must be flat" in issue for issue in issues)
    assert all("archived/ is the only special top-level bucket" in issue for issue in issues)


def test_shared_usr_dataset_layout_check_allows_flat_roots_and_archived_bucket(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "datasets" / "densegen_demo_sampling_baseline" / "records.parquet",
        "placeholder\n",
    )
    _write(
        tmp_path
        / "src"
        / "dnadesign"
        / "usr"
        / "datasets"
        / "archived"
        / "densegen"
        / "demo_sampling_baseline"
        / "records.parquet",
        "placeholder\n",
    )

    assert _find_shared_usr_dataset_layout_issues(tmp_path) == []


def test_main_fails_for_broken_relative_link_in_root_sor_doc(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(tmp_path / "ARCHITECTURE.md", "[broken](docs/missing.md)\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_tool_readme_banner_check_flags_missing_or_non_svg_banners(tmp_path: Path) -> None:
    tool_root = tmp_path / "src" / "dnadesign"
    _write(tool_root / "alpha" / "README.md", "## Alpha\n\nNo banner.\n")
    _write(tool_root / "beta" / "README.md", "## Beta\n\n![Beta banner](images/beta-banner.png)\n")

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert any("alpha/README.md" in issue and "missing top banner image" in issue for issue in issues)
    assert any("beta/README.md" in issue and "must target a local .svg asset" in issue for issue in issues)


def test_tool_readme_banner_check_accepts_existing_local_svg_banner(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "## Alpha\n\n![Alpha banner](assets/alpha-banner.svg)\n\nCompact subtitle.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert issues == []


def test_tool_readme_banner_check_rejects_nonstandard_banner_dimensions(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="420" viewBox="0 0 1600 420"></svg>\n',
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert any("1200x180 SVG contract" in issue for issue in issues)


def test_tool_readme_structure_check_requires_banner_as_first_non_empty_line(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "## Alpha\n\n![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("first non-empty line must be the banner image line" in issue for issue in issues)


def test_tool_readme_structure_check_rejects_heading_immediately_after_banner(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alpha-banner.svg)\n\n## Alpha\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("line after the banner must be narrative text" in issue for issue in issues)


def test_tool_readme_structure_check_requires_top_level_markdown_doc_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("top section must include a local markdown link" in issue for issue in issues)


def test_tool_readme_structure_check_accepts_banner_narrative_and_docs_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Short narrative overview.",
                "",
                "## Documentation",
                "",
                "See [docs index](../../../docs/README.md) for workflows and references.",
                "",
                "## Usage",
                "",
                "Run alpha.",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert issues == []


def test_tool_readme_structure_check_rejects_multi_paragraph_intro(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "First narrative paragraph.",
                "",
                "Second narrative paragraph belongs in deeper docs.",
                "",
                "## Documentation",
                "",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("intro after the banner must be one paragraph" in issue for issue in issues)


def test_tool_readme_structure_check_rejects_self_referential_intro(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Alpha is the analysis package in `dnadesign`.",
                "",
                "## Documentation",
                "",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("avoid self-referential package/layer-in-dnadesign wording" in issue for issue in issues)


def test_tool_readme_structure_check_requires_documentation_heading(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Alpha scores short sequence examples.",
                "",
                "## Start here",
                "",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("first heading after the intro must be '## Documentation'" in issue for issue in issues)


def test_tool_readme_structure_check_rejects_overlong_tool_readmes(tmp_path: Path) -> None:
    body_lines = [
        "![Alpha banner](assets/alpha-banner.svg)",
        "",
        "Short narrative.",
        "",
        "## Documentation",
        "",
        "[Docs](docs/README.md)",
    ]
    body_lines.extend(f"Extra line {idx}." for idx in range(40))
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "README.md", "\n".join(body_lines) + "\n")
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("top-level tool README has" in issue for issue in issues)


def test_tool_readme_structure_check_requires_docs_index_first_when_present(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Short narrative.",
                "",
                "## Documentation",
                "",
                "[Repository docs](../../../docs/README.md)",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("first local markdown link must point to the tool docs index" in issue for issue in issues)


def test_root_docs_entrypoint_check_requires_docs_index_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "Use the docs index.",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert any("must include a markdown link to docs/README.md" in issue for issue in issues)


def test_root_docs_entrypoint_check_accepts_banner_with_docs_index_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert issues == []


def test_root_docs_entrypoint_check_rejects_plain_text_paths_without_links(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "Use docs/README.md as the docs entrypoint.",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert any("must include a markdown link to docs/README.md" in issue for issue in issues)


def test_root_docs_entrypoint_check_rejects_bannerless_readme_without_docs_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "Use the docs index.",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert any("must include a markdown link to docs/README.md" in issue for issue in issues)


def test_deprecated_docs_entrypoint_check_flags_start_here_file(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "start-here.md", "## Start Here\n")

    issues = _find_deprecated_docs_entrypoint_issues(tmp_path)

    assert any("docs/start-here.md" in issue and "deprecated" in issue for issue in issues)


def test_deprecated_docs_entrypoint_check_flags_start_here_links(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Start here](docs/start-here.md)",
                "",
            ]
        ),
    )

    issues = _find_deprecated_docs_entrypoint_issues(tmp_path)

    assert any("must not link to docs/start-here.md" in issue for issue in issues)


def test_entrypoint_local_path_link_check_flags_local_literal_paths(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "setup" / "installation.md", "## Installation\n")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
                "Read `docs/setup/installation.md` before running commands.",
                "",
            ]
        ),
    )

    issues = _find_entrypoint_local_path_literal_issues(tmp_path)

    assert any("local path literal" in issue and "docs/setup/installation.md" in issue for issue in issues)


def test_entrypoint_local_path_link_check_allows_hyperlinked_local_paths(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
                "Read [installation guide](docs/setup/installation.md) before running commands.",
                "",
            ]
        ),
    )

    issues = _find_entrypoint_local_path_literal_issues(tmp_path)

    assert issues == []


def test_agents_path_reference_check_flags_missing_scoped_paths(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "AGENTS.md",
        "- Default config: `src/dnadesign/cruncher/workspaces/missing/config.yaml`\n",
    )

    issues = _find_agents_path_reference_issues(tmp_path)

    assert any("src/dnadesign/cruncher/workspaces/missing/config.yaml" in issue for issue in issues)


def test_agents_path_reference_check_allows_existing_and_non_path_spans(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "cruncher" / "workspaces" / "demo" / "configs" / "config.yaml", "{}\n")
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "AGENTS.md",
        "\n".join(
            [
                "- Default config: `workspaces/demo/configs/config.yaml`",
                "- Repo-root config: `src/dnadesign/cruncher/workspaces/demo/configs/config.yaml`",
                "- This repo intentionally has no `./scripts/agent-verify`.",
                "- Use command `uv run cruncher --help`.",
                "- Template path: `workspaces/<id>/configs/config.yaml`.",
                "",
            ]
        ),
    )

    issues = _find_agents_path_reference_issues(tmp_path)

    assert issues == []


def test_densegen_docs_language_check_flags_canonical_term(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "README.md", "This is the canonical densegen guide.\n")

    issues = _find_densegen_disallowed_term_issues(tmp_path)

    assert any("term 'canonical'" in issue for issue in issues)


def test_densegen_docs_language_check_accepts_plain_language(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "README.md", "DenseGen guide.\n")
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "docs" / "tutorials" / "demo.md",
        "## Demo\n\nUse this tutorial to run the workflow.\n",
    )

    issues = _find_densegen_disallowed_term_issues(tmp_path)

    assert issues == []


def test_runbook_demo_snippet_check_flags_missing_shell_and_yaml_comments(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "tutorials" / "demo.md",
        "\n".join(
            [
                "## Demo",
                "",
                "```bash",
                "uv run alpha do-work",
                "```",
                "",
                "```yaml",
                "alpha:",
                "  enabled: true",
                "```",
                "",
            ]
        ),
    )

    issues = _find_runbook_demo_snippet_issues(tmp_path)

    assert any("command in shell block needs an explanatory comment" in issue for issue in issues)
    assert any("yaml key/value in runbook/demo snippets needs a right-side inline comment" in issue for issue in issues)


def test_runbook_demo_snippet_check_accepts_commented_shell_and_yaml_blocks(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "tutorials" / "demo.md",
        "\n".join(
            [
                "## Demo",
                "",
                "```bash",
                "# Run the demo command.",
                "uv run alpha do-work",
                "```",
                "",
                "```yaml",
                "alpha:",
                "  enabled: true  # Toggle demo mode.",
                "```",
                "",
            ]
        ),
    )

    issues = _find_runbook_demo_snippet_issues(tmp_path)

    assert issues == []


def test_main_fails_when_root_sor_doc_missing_required_metadata(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(tmp_path / "ARCHITECTURE.md", "# ARCHITECTURE\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_root_sor_doc_missing_type_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_root_sor_doc_last_verified_is_stale(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        "# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )

    rc = main(["--repo-root", str(tmp_path), "--max-sor-age-days", "30"])
    assert rc == 1


def test_main_fails_when_docs_index_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "## Documentation Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_when_docs_index_has_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        f"## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_start_here_doc_is_present(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        f"## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "docs" / "start-here.md", "## Start Here\n\nPick a path.\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_docs_index_last_verified_is_stale(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        "## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path), "--max-sor-age-days", "30"])
    assert rc == 1


def test_main_fails_when_selected_runbook_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "setup" / "installation.md",
        "## Installation\n\nRun setup.\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_selected_runbook_last_verified_is_stale(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "setup" / "installation.md",
        "## Installation\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path), "--max-sor-age-days", "30"])
    assert rc == 1


def test_main_passes_when_selected_runbook_has_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "setup" / "installation.md",
        f"## Installation\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_operations_runbook_docs_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "operations" / "README.md",
        "## Ops orchestration index\n\nMissing metadata.\n",
    )
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "## Orchestration runbooks\n\nMissing metadata.\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_find_operational_runbook_path_issues_flags_repo_root_runbook(tmp_path: Path) -> None:
    _write(
        tmp_path / "stress_ethanol_cipro.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: study_stress_ethanol_cipro",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 08:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )
    _git_init(tmp_path)
    _git_add(tmp_path, "stress_ethanol_cipro.yaml")

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert any("operational runbook path is outside allowed locations" in issue for issue in issues)


def test_find_operational_runbook_path_issues_rejects_malformed_tracked_yaml(tmp_path: Path) -> None:
    _write(tmp_path / "broken.yaml", "runbook:\n  workflow_id: [broken\n")
    _git_init(tmp_path)
    _git_add(tmp_path, "broken.yaml")

    with pytest.raises(ValueError, match="operational runbook yaml is invalid"):
        _find_operational_runbook_path_issues(tmp_path)


def test_find_operational_runbook_path_issues_ignores_untracked_yaml_noise_in_git_repo(tmp_path: Path) -> None:
    _write(
        tmp_path / "stress_ethanol_cipro.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: study_stress_ethanol_cipro",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 08:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "scratch" / "nested" / "noise.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: generated_noise",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
            ]
        )
        + "\n",
    )
    _git_init(tmp_path)
    _git_add(tmp_path, "stress_ethanol_cipro.yaml")

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert any("stress_ethanol_cipro.yaml" in issue for issue in issues)
    assert not any("scratch/nested/noise.yaml" in issue for issue in issues)


def test_find_operational_runbook_path_issues_allows_packaged_presets(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "runbooks" / "presets" / "densegen_demo.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: study_stress_ethanol_cipro",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 08:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert issues == []


def test_find_operational_runbook_path_issues_allows_workspace_runbooks_dir(tmp_path: Path) -> None:
    _write(
        tmp_path / "workspace" / "outputs" / "logs" / "ops" / "runbooks" / "densegen_demo.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: densegen_demo",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/densegen_demo",
            ]
        )
        + "\n",
    )

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert issues == []


def test_find_operational_runbook_path_issues_skips_generated_output_yaml_noise(tmp_path: Path) -> None:
    _write(
        tmp_path / "workspace" / "outputs" / "usr_datasets" / "registry.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: generated_noise",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
            ]
        )
        + "\n",
    )

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert issues == []


def test_find_packaged_runbook_variant_issues_flags_duration_suffixed_preset(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "runbooks" / "presets" / "densegen_demo_with_notify_6h.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: densegen_demo_with_notify_6h",
                "  workflow_id: densegen_batch_with_notify",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/densegen_demo_with_notify_6h",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 06:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )

    issues = _find_packaged_runbook_variant_issues(tmp_path)

    assert any("duration-suffixed operational variants are not allowed in presets" in issue for issue in issues)


def test_find_packaged_runbook_variant_issues_allows_base_preset_name(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "runbooks" / "presets" / "densegen_demo_with_notify.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: densegen_demo_with_notify",
                "  workflow_id: densegen_batch_with_notify",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/densegen_demo_with_notify",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 08:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )

    issues = _find_packaged_runbook_variant_issues(tmp_path)

    assert issues == []


def test_find_shared_utils_path_issues_flags_top_level_utils_package(tmp_path: Path) -> None:
    disallowed_utils_path = tmp_path / "src" / "dnadesign" / "utils"
    disallowed_utils_path.mkdir(parents=True, exist_ok=True)

    issues = _find_shared_utils_path_issues(tmp_path)

    assert any("shared utils package is not allowed" in issue for issue in issues)


def test_find_shared_utils_path_issues_allows_tool_local_utils(tmp_path: Path) -> None:
    allowed_tool_utils_path = tmp_path / "src" / "dnadesign" / "densegen" / "src" / "utils"
    allowed_tool_utils_path.mkdir(parents=True, exist_ok=True)

    issues = _find_shared_utils_path_issues(tmp_path)

    assert issues == []


def test_docs_checks_reuses_ops_path_policy_contract_constants() -> None:
    from dnadesign.devtools.docs import checks as docs_checks

    assert docs_checks.TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES == REPO_TRANSIENT_OPERATIONAL_DIR_NAMES
    assert docs_checks.OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES[0] == PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR


def test_find_transient_operational_artifact_path_issues_flags_repo_root_codex_tmp(tmp_path: Path) -> None:
    _write(tmp_path / ".codex_tmp" / "audit_notify" / "records.parquet", "placeholder\n")

    issues = _find_transient_operational_artifact_path_issues(tmp_path)

    assert any("transient operational artifact directory is not allowed at repo root" in issue for issue in issues)


def test_find_transient_operational_artifact_path_issues_allows_workspace_nested_temp_dirs(tmp_path: Path) -> None:
    _write(
        tmp_path
        / "src"
        / "dnadesign"
        / "densegen"
        / "workspaces"
        / "study"
        / "outputs"
        / "tmp"
        / ".codex_tmp"
        / "state.json",
        "{}\n",
    )

    issues = _find_transient_operational_artifact_path_issues(tmp_path)

    assert issues == []


def test_main_fails_when_repo_root_contains_transient_operational_dir(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(tmp_path / ".tmp_ops" / "scratch.log", "placeholder\n")

    rc = main(["--repo-root", str(tmp_path)])

    assert rc == 1


def test_find_stale_overlay_guard_term_issues_flags_old_ops_guard_terms(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
                "",
                "Use densegen-overlay-guard with densegen.overlay_guard.namespace.",
                "",
            ]
        )
        + "\n",
    )

    issues = _find_stale_overlay_guard_term_issues(tmp_path)

    assert any("densegen-overlay-guard" in issue for issue in issues)
    assert any("densegen.overlay_guard.namespace" in issue for issue in issues)


def test_find_stale_overlay_guard_term_issues_accepts_usr_overlay_guard_terms(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
                "",
                "Use usr-overlay-guard with densegen.overlay_guard.overlay_namespace.",
                "",
            ]
        )
        + "\n",
    )

    issues = _find_stale_overlay_guard_term_issues(tmp_path)

    assert issues == []


def test_main_fails_when_exec_plan_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "docs" / "exec-plans" / "active" / "example.md", "# Exec plan\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_missing_required_living_sections(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "- [ ] (2026-02-18 10:00Z) pending",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_for_exec_plan_with_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "## Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "### Purpose / Big Picture",
                "Purpose.",
                "",
                "### Progress",
                "- [ ] (2026-02-18 10:00Z) pending",
                "",
                "### Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "### Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "### Outcomes & Retrospective",
                "Pending.",
                "",
                "### Context and Orientation",
                "Context.",
                "",
                "### Plan of Work",
                "Plan.",
                "",
                "### Concrete Steps",
                "Run command.",
                "",
                "### Validation and Acceptance",
                "Validate behavior.",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_exec_plan_progress_has_no_checklist_items(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "Progress narrative only.",
                "",
                "## Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "## Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "## Outcomes & Retrospective",
                "Pending.",
                "",
                "## Context and Orientation",
                "Context.",
                "",
                "## Plan of Work",
                "Plan.",
                "",
                "## Concrete Steps",
                "Run command.",
                "",
                "## Validation and Acceptance",
                "Validate behavior.",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_has_checklist_outside_progress(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "- [ ] pending",
                "",
                "## Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "## Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "## Outcomes & Retrospective",
                "Pending.",
                "",
                "## Context and Orientation",
                "Context.",
                "",
                "## Plan of Work",
                "Plan.",
                "",
                "## Concrete Steps",
                "Run command.",
                "",
                "## Validation and Acceptance",
                "- [ ] run tests",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_progress_checklist_lacks_timestamp(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "- [ ] pending",
                "",
                "## Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "## Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "## Outcomes & Retrospective",
                "Pending.",
                "",
                "## Context and Orientation",
                "Context.",
                "",
                "## Plan of Work",
                "Plan.",
                "",
                "## Concrete Steps",
                "Run command.",
                "",
                "## Validation and Acceptance",
                "Validate behavior.",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_for_valid_links(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        f"## x\n\n**Owner:** maintainers\n**Last verified:** {today}\n\n"
        "[guide](./guide.md)\n[#anchor](#x)\n[site](https://example.com)\n",
    )
    _write(tmp_path / "docs" / "guide.md", "## Guide\n")
    _write(tmp_path / "README.md", "[docs](docs/README.md)\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n[docs](docs/guide.md)\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_readme_tool_catalog_missing_repo_tool(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_readme_tool_catalog_does_not_require_studies_row(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "studies" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "README.md", "# aligner\n")
    _write(tmp_path / "src" / "dnadesign" / "studies" / "README.md", "# studies\n")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    assert _find_readme_tool_catalog_issues(tmp_path) == []


def test_main_fails_when_readme_tool_catalog_row_has_too_few_columns(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_catalog_missing_coverage_column(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_when_readme_tool_catalog_matches_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", f"## Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "    - component_id: notify",
                "      name: notify",
                "      paths:",
                "        - src/dnadesign/notify/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_readme_tool_link_does_not_match_expected_path(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/docs) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_link_target_directory_is_missing(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_coverage_link_component_mismatches_tool(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_codecov_components_do_not_cover_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_codecov_component_default_rules_are_missing(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_public_interface_docs_use_absolute_paths(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "docs" / "tutorials" / "demo.md",
        "Use `uv run cruncher catalog export-densegen --densegen-workspace /tmp/demo`.\n",
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**densegen**](src/dnadesign/densegen/README.md) | densegen tool | "
                "[Codecov](https://codecov.io/gh/example/repo?component=densegen) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: densegen",
                "      name: densegen",
                "      paths:",
                "        - src/dnadesign/densegen/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_public_interface_docs_use_internal_source_inreach(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "docs" / "howto" / "handoff.md",
        "Call `python -m dnadesign.cruncher.src.cli.app` directly.\n",
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**densegen**](src/dnadesign/densegen/README.md) | densegen tool | "
                "[Codecov](https://codecov.io/gh/example/repo?component=densegen) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: densegen",
                "      name: densegen",
                "      paths:",
                "        - src/dnadesign/densegen/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_public_interface_doc_contract_includes_maintainer_and_runbook_routers(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "dev" / "README.md", "Call `python -m dnadesign.cruncher.src.cli.app`.\n")
    _write(tmp_path / "docs" / "runbooks" / "README.md", "Use `/tmp/local-runbook.yaml` for scratch work.\n")

    issues = _find_public_interface_doc_contract_issues(tmp_path)

    assert any("docs/dev/README.md" in issue and "internal source inreach" in issue for issue in issues)
    assert any("docs/runbooks/README.md" in issue and "absolute filesystem path token" in issue for issue in issues)


def test_public_interface_doc_contract_includes_top_level_tool_readmes(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "opal" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "README.md",
        "Call `python -m dnadesign.cruncher.src.cli.app` from `/tmp/opal-demo`.\n",
    )

    issues = _find_public_interface_doc_contract_issues(tmp_path)

    assert any("src/dnadesign/opal/README.md" in issue and "internal source inreach" in issue for issue in issues)
    assert any(
        "src/dnadesign/opal/README.md" in issue and "absolute filesystem path token" in issue for issue in issues
    )


def test_broken_link_check_includes_top_level_tool_readmes(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", f"## Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "opal" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "README.md",
        "![opal banner](assets/opal-banner.svg)\n\nOPAL narrative.\n\n## Documentation\n\n[Missing](docs/missing.md)\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "opal" / "assets" / "opal-banner.svg", VALID_TOOL_BANNER_SVG)
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**opal**](src/dnadesign/opal/README.md) | opal tool | "
                "[Codecov](https://codecov.io/gh/example/repo?component=opal) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: opal",
                "      name: opal",
                "      paths:",
                "        - src/dnadesign/opal/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])

    assert rc == 1


def test_main_passes_when_codecov_components_match_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", f"## Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "    - component_id: notify",
                "      name: notify",
                "      paths:",
                "        - src/dnadesign/notify/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_docs_root_heading_style_check_flags_level_one_or_repeated_level_two_headings(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "README.md",
        "\n".join(
            [
                "# Documentation Index",
                "",
                "## One",
                "## Two",
                "",
            ]
        ),
    )

    issues = _find_docs_root_heading_style_issues(tmp_path)

    assert any("must start with '## '" in issue for issue in issues)
    assert any("use a single level-2 heading" in issue for issue in issues)


def test_docs_root_heading_style_check_accepts_level_two_title_and_lower_sections(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "README.md",
        "\n".join(
            [
                "## Documentation Index",
                "",
                "### Use this index",
                "",
                "Text.",
                "",
            ]
        ),
    )

    issues = _find_docs_root_heading_style_issues(tmp_path)

    assert issues == []


def test_broken_links_check_flags_missing_markdown_anchor(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "source.md"
    target = tmp_path / "docs" / "target.md"
    _write(source, "[missing](./target.md#not-here)\n")
    _write(target, "## Present Heading\n")

    broken = _find_broken_links([source, target])

    assert any("anchor 'not-here'" in issue_link for _, issue_link in broken)


def test_tool_docs_metadata_check_flags_missing_owner_and_last_verified(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_docs_metadata_issues(tmp_path, max_age_days=90)

    assert any("missing '**Owner:**' metadata field." in issue for issue in issues)
    assert any("missing '**Last verified:**' metadata field." in issue for issue in issues)


def test_tool_docs_metadata_check_accepts_valid_owner_and_last_verified(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md",
        f"## Alpha docs\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    issues = _find_tool_docs_metadata_issues(tmp_path, max_age_days=90)

    assert issues == []


def test_cross_tool_doc_metadata_check_flags_missing_semantic_fields(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "README.md",
        "\n".join(
            [
                "## Ops orchestration index",
                "",
                "**Type:** route",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
                "",
                "Missing plane and artifact metadata.",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert any("missing '**Plane:**'" in issue for issue in issues)
    assert any("missing '**Owner-boundary:**'" in issue for issue in issues)
    assert any("missing '**Entry artifact:**'" in issue for issue in issues)
    assert any("missing '**Exit artifact:**'" in issue for issue in issues)


def test_cross_tool_doc_metadata_check_flags_missing_registry_fields(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert any("missing '**Registry-id:**'" in issue for issue in issues)
    assert any("missing '**Summary:**'" in issue for issue in issues)
    assert any("missing '**Execution-kind:**'" in issue for issue in issues)
    assert any("missing '**Status-kind:**'" in issue for issue in issues)


def test_cross_tool_doc_metadata_check_accepts_expected_contract_values(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "README.md",
        "\n".join(
            [
                "## Ops orchestration index",
                "",
                "**Type:** route",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** batch orchestration intent",
                "**Exit artifact:** authoritative ops contract",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert issues == []


def test_cross_tool_doc_metadata_check_accepts_registry_fields_for_runbook_docs(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic batch orchestration contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert issues == []


def test_runbook_catalog_check_flags_missing_registered_doc_entries(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic batch orchestration contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic batch orchestration contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    hpc_sync_doc = tmp_path / "src" / "dnadesign" / "usr" / "docs" / "operations" / "sync" / "hpc-agent-flow.md"
    _write(
        hpc_sync_doc,
        "\n".join(
            [
                "## USR HPC Sync Flow",
                "",
                "**Type:** runbook",
                "**Plane:** data-plane",
                "**Owner-boundary:** usr",
                "**Entry artifact:** sync intent",
                "**Exit artifact:** synchronized dataset",
                "**Registry-id:** usr.data-plane.hpc-sync",
                "**Summary:** HPC and local sync flow.",
                "**Execution-kind:** iterative",
                "**Status-kind:** usr-sync-audit",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("missing registry metadata sidecar" in issue for issue in issues)
    assert any("src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.registry.yaml" in issue for issue in issues)


def test_runbook_catalog_check_flags_metadata_drift_against_owner_local_doc(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic batch orchestration contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any(
        "Summary for docs/operations/orchestration/runbooks.md must match owner-local metadata" in issue
        for issue in issues
    )


def test_runbook_catalog_check_accepts_matching_owner_local_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    summary = "Deterministic control-plane runbook contract."
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                f"**Summary:** {summary}",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary=summary,
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert issues == []


def test_runbook_catalog_check_flags_stale_generated_procedure_section(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_runbook_catalog_readme(
        tmp_path,
        procedure_section="stale manually edited procedure section",
        tool_source_section=_empty_tool_source_section(),
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("cross-tool procedures section is stale" in issue for issue in issues)


def test_runbook_catalog_check_flags_stale_generated_tool_source_section(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    ops_docs = tmp_path / "src" / "dnadesign" / "ops" / "docs" / "README.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write(
        ops_docs,
        "\n".join(
            [
                "## ops docs",
                "",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_tool_source_metadata(
        ops_docs,
        catalog_order=1,
        tool="ops",
        summary="Control-plane docs.",
        keywords=["control-plane", "runbooks"],
    )
    _write_runbook_catalog_readme(
        tmp_path,
        procedure_section="_placeholder_",
        tool_source_section="_placeholder_",
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )
    _write_runbook_catalog_readme(
        tmp_path,
        procedure_section=render_catalog_procedure_section(load_runbook_catalog(repo_root=tmp_path)),
        tool_source_section="stale manually edited tool-source section",
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("tool docs section is stale" in issue for issue in issues)


def test_runbook_catalog_check_flags_missing_progress_surface_glossary_entry(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `usr-sync-audit` | Sync drift review. | Inspect the sync audit. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("missing status surface glossary entry for 'ops-audit-json'" in issue for issue in issues)


def test_runbook_catalog_check_uses_status_registry_inventory_for_glossary(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "providers" / "builtin" / "status.registry.yaml",
        "\n".join(
            [
                "version: 1",
                "provider_id: builtin.ops",
                "entries:",
                "  - status_kind: ops-audit-json",
                "    owner_boundary: ops",
                "    observes_plane: control",
                "    provider_ref: dnadesign.ops.providers.builtin.status_provider:provide_ops_audit_status",
                "    description: Read one orchestration audit JSON.",
                "    surface_type: orchestration_audit",
                "    cost_class: cheap",
                "    summary_scope: workspace",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "latentdna" / "ops" / "status.registry.yaml",
        "\n".join(
            [
                "version: 1",
                "provider_id: latentdna.workspace-status",
                "entries:",
                "  - status_kind: latentdna-workspace-snapshot",
                "    owner_boundary: latentdna",
                "    observes_plane: data",
                "    provider_ref: dnadesign.latentdna.ops.status_providers:provide_snapshot",
                "    description: Read one LatentDNA workspace snapshot.",
                "    surface_type: artifact_catalog",
                "    cost_class: cheap",
                "    summary_scope: workspace",
                "",
            ]
        ),
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("missing status surface glossary entry for 'latentdna-workspace-snapshot'" in issue for issue in issues)


def test_ops_deprecated_semantics_check_flags_legacy_terms(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Ops runbook",
                "",
                "Use `densegen_batch_with_notify_slack`.",
                "",
                "The precedents surface remains available.",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "docs" / "studies" / "README.md",
        "Use infer_local_runtime and notify_profile_doctor in ops.study.yaml.\n",
    )
    _write(
        tmp_path / "docs" / "studies" / "stress_ethanol_cipro_growth" / "contracts" / "preflight.md",
        "Read notify.profile.*.details.setup_command after infer_validate_config.\n",
    )

    issues = _find_ops_deprecated_semantics_issues(tmp_path)

    assert any("with_notify_slack" in issue for issue in issues)
    assert any("precedents" in issue for issue in issues)
    assert any("infer_local_runtime" in issue for issue in issues)
    assert any("notify_profile_doctor" in issue for issue in issues)
    assert any("details.setup_command" in issue for issue in issues)


def test_study_execution_source_drift_check_flags_pipeline_only_claims(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "studies" / "README.md",
        "Use pipeline.yaml as the only source for real Construct, Infer, and runbook paths.\n",
    )
    _write(
        tmp_path / "docs" / "studies" / "stress_ethanol_cipro_growth" / "contracts" / "preflight.md",
        "pipeline.yaml remains the only valid source for exact execution surfaces.\n",
    )

    issues = _find_study_execution_source_drift_issues(tmp_path)

    assert any("docs/studies/README.md" in issue for issue in issues)
    assert any("preflight.md" in issue for issue in issues)


def test_legacy_contract_surface_docs_check_flags_repo_root_contract_references(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "README.md", "## Docs\n\nUse `dnadesign._contracts` and `src/dnadesign/usr_roots.py`.\n")

    issues = _find_legacy_contract_surface_doc_issues(tmp_path)

    assert any("dnadesign._contracts" in issue for issue in issues)
    assert any("src/dnadesign/usr_roots.py" in issue for issue in issues)
