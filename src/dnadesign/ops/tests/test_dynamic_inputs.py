"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_dynamic_inputs.py

Focused tests for metadata-driven progress input parsing and rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.ops.cli.dynamic_inputs import parse_status_input_tokens, render_progress_show_command
from dnadesign.ops.status.models import InputFieldSpec


def test_parse_status_input_tokens_accepts_declared_flag_and_input_forms() -> None:
    input_schema = (
        InputFieldSpec(
            name="study_dir",
            cli_flag="--study-dir",
            placeholder="<study-dir>",
            summary="Checked-in study directory.",
            type="path",
            required=False,
        ),
        InputFieldSpec(
            name="scope",
            cli_flag="--scope",
            placeholder="<scope>",
            summary="Preflight scope.",
            type="enum",
            required=False,
            default="next",
            choices=("next", "full"),
        ),
    )

    resolved = parse_status_input_tokens(
        extra_args=("--study-dir", "docs/studies/current"),
        input_items=("scope=full",),
        input_schema=input_schema,
    )

    assert resolved == {
        "study_dir": "docs/studies/current",
        "scope": "full",
    }


def test_parse_status_input_tokens_rejects_duplicate_input_across_flag_and_escape_hatch() -> None:
    input_schema = (
        InputFieldSpec(
            name="scope",
            cli_flag="--scope",
            placeholder="<scope>",
            summary="Preflight scope.",
            type="enum",
            required=False,
            default="next",
            choices=("next", "full"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate progress input: --scope"):
        parse_status_input_tokens(
            extra_args=("--scope", "next"),
            input_items=("scope=full",),
            input_schema=input_schema,
        )


def test_render_progress_show_command_preserves_metadata_placeholders() -> None:
    rendered = render_progress_show_command(
        registry_id="usr.data-plane.promoter-study-preflight",
        required_inputs=(
            InputFieldSpec(
                name="study_dir",
                cli_flag="--study-dir",
                placeholder="<study-dir>",
                summary="Checked-in study directory.",
                type="path",
                required=True,
            ),
        ),
    )

    assert rendered == "uv run ops progress show usr.data-plane.promoter-study-preflight --study-dir <study-dir>"
