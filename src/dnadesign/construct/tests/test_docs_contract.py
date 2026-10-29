"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/test_docs_contract.py

Documentation placement contracts for construct cross-tool workflow routing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.construct import load_job_config, summarize_linear_ssdna_composition


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def test_construct_docs_route_to_shared_source_of_truth_runbook() -> None:
    top_readme = _read("src/dnadesign/construct/README.md")
    readme = _read("src/dnadesign/construct/docs/README.md")
    index_doc = _read("src/dnadesign/construct/docs/index.md")
    config_doc = _read("src/dnadesign/construct/docs/reference/config.md")
    outputs = _read("src/dnadesign/construct/docs/reference/outputs.md")
    template_contexts = _read("src/dnadesign/construct/docs/reference/template-contexts.md")
    workspaces = _read("src/dnadesign/construct/workspaces/README.md")
    source_of_truth_workspace = _read(
        "src/dnadesign/construct/workspaces/demo_anchor_template_shared_dataset/README.md"
    )

    token = "../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md"
    multi_source_token = "../../seqledger/docs/operations/assembly/multi-source-shared-dataset.md"
    feature_matrix_token = "../../seqledger/docs/operations/promoter/characterization-feature-matrix.md"
    assert token in readme
    assert multi_source_token in readme
    assert feature_matrix_token in readme
    assert "docs/README.md" in top_readme
    assert "docs/getting-started.md" in top_readme
    assert "workspaces/README.md" in top_readme
    assert "docs/reference/template-contexts.md" in top_readme
    assert "../../../docs/README.md" in top_readme
    assert "reference/template-contexts.md" in readme
    assert "template-contexts.md" in config_doc
    assert "../../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md" in outputs
    assert "construct__anchor_start" in outputs
    assert "construct__anchor_start" in template_contexts
    assert "demo_anchor_template_shared_dataset" in readme
    assert "demo_anchor_template_shared_dataset" in workspaces
    assert "promoter/characterization-feature-matrix.md" in source_of_truth_workspace
    assert "(README.md)" in index_doc
    assert "(getting-started.md)" in index_doc


def test_construct_cli_reference_binds_the_explicit_usr_root_across_execution() -> None:
    cli = _read("src/dnadesign/construct/docs/reference/cli.md")

    for command in (
        "construct run --config <path>",
        "construct validate config --config <path>",
        "construct workspace validate-project",
        "construct workspace run-project",
    ):
        line = next(line for line in cli.splitlines() if command in line)
        assert "--seqledger-root <absolute-path>" in line
    assert "existing directory that is not a symbolic link" in cli


def test_config_reference_example_uses_the_public_job_schema(tmp_path: Path) -> None:
    text = _read("src/dnadesign/construct/docs/reference/config.md")
    example = text.split("```yaml\n", 1)[1].split("```", 1)[0]
    config_path = tmp_path / "example.yaml"
    config_path.write_text(example, encoding="utf-8")

    config, _ = load_job_config(config_path)

    assert config.job.input.source.kind == "usr"
    assert config.job.output.target.kind == "usr"


def test_composition_reference_example_produces_the_documented_length(tmp_path: Path) -> None:
    text = _read("src/dnadesign/construct/docs/reference/linear-ssdna-composition.md")
    example = text.split("```yaml\n", 1)[1].split("```", 1)[0]
    config_path = tmp_path / "composition.yaml"
    config_path.write_text(example, encoding="utf-8")

    result = summarize_linear_ssdna_composition(config_path)

    assert result.unit_count == 1
    assert result.expanded_copy_count == 1
    assert result.sequence_length == 12
