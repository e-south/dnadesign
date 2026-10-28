"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/features.py

Explicit maintenance commands for persisted feature payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path

import typer

from ...features.aliases import compact_feature_sidecars, migrate_feature_payloads
from ..common import raise_cli_error


def register(app: typer.Typer) -> None:
    features = typer.Typer(no_args_is_help=True, help="Migrate or compact saved feature payloads.")
    app.add_typer(features, name="features")

    @features.command("migrate")
    def migrate(
        root: Path = typer.Option(..., help="Dataset root."),
        dataset: str = typer.Option(..., help="Dataset id."),
    ) -> None:
        """Verify and convert single-file payloads to committed parts."""
        try:
            result = migrate_feature_payloads(dataset_root=root, dataset_id=dataset)
        except (ValueError, OSError) as exc:
            raise_cli_error(exc)
        else:
            typer.echo(json.dumps(result, sort_keys=True))

    @features.command("compact")
    def compact(
        root: Path = typer.Option(..., help="Dataset root."),
        dataset: str = typer.Option(..., help="Dataset id."),
        prune_unreferenced: bool = typer.Option(False, help="Also remove payloads without current aliases."),
    ) -> None:
        """Combine committed parts and remove orphaned temporary payloads."""
        try:
            result = compact_feature_sidecars(
                dataset_root=root, dataset_id=dataset, prune_unreferenced=prune_unreferenced
            )
        except (ValueError, OSError) as exc:
            raise_cli_error(exc)
        else:
            typer.echo(json.dumps(result, sort_keys=True))
