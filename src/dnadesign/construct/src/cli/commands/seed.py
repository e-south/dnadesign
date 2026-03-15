"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/seed.py

Curated construct bootstrap commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from dnadesign.usr import default_usr_root

from ...errors import ConstructError
from ...seed import bootstrap_promoter_swap_demo, import_seed_manifest

seed_app = typer.Typer(
    no_args_is_help=True,
    help="Seed curated or manifest-driven local USR datasets for construct workflows.",
)


@seed_app.command("import-manifest")
def import_manifest(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        exists=True,
        readable=True,
        help="Manifest YAML describing one or more input/template datasets to materialize into USR.",
    ),
    root: Path = typer.Option(
        default_usr_root(),
        "--root",
        help="Target local USR datasets root. Passing src/dnadesign/usr is also accepted and normalized.",
    ),
) -> None:
    try:
        result = import_seed_manifest(root=root, manifest=manifest)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1) from exc

    typer.echo(f"seed_root: {result.root}")
    typer.echo(f"manifest_id: {result.manifest_id}")
    for dataset in result.datasets:
        typer.echo(f"dataset: {dataset.dataset}")
        typer.echo(f"dataset_dir: {result.root / dataset.dataset}")
        for entry in dataset.entries:
            typer.echo(
                "record: "
                f"label={entry.label} "
                f"record_id={entry.record_id} "
                f"role={entry.role} "
                f"topology={entry.topology} "
                f"length_bp={len(entry.sequence)}"
            )


@seed_app.command("promoter-swap-demo")
def promoter_swap_demo(
    root: Path = typer.Option(
        default_usr_root(),
        "--root",
        help="Target local USR datasets root. Passing src/dnadesign/usr is also accepted and normalized.",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional manifest YAML path to write record ids and slot metadata.",
    ),
) -> None:
    try:
        result = bootstrap_promoter_swap_demo(root=root, manifest=manifest)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1) from exc

    typer.echo(f"seed_root: {result.root}")
    typer.echo(f"anchor_dataset: {result.anchor_dataset}")
    typer.echo(f"anchor_dataset_dir: {result.root / result.anchor_dataset}")
    for entry in result.anchor_entries:
        typer.echo(f"anchor: label={entry.label} record_id={entry.record_id} length_bp={len(entry.sequence)}")
    typer.echo(f"template_dataset: {result.template_dataset}")
    typer.echo(f"template_dataset_dir: {result.root / result.template_dataset}")
    for entry in result.template_entries:
        typer.echo(
            "template: "
            f"label={entry.label} "
            f"record_id={entry.record_id} "
            f"length_bp={len(entry.sequence)} "
            f"topology={entry.topology}"
        )
    for slot in result.slots:
        typer.echo(
            f"slot: name={slot.slot} start={slot.start} end={slot.end} expected_length_bp={slot.end - slot.start}"
        )
    if result.manifest_path is not None:
        typer.echo(f"manifest: {result.manifest_path}")

    legacy_paths = [
        result.root / "construct" / "promoter_swap_controls_demo",
        result.root / "construct" / "promoter_swap_templates_demo",
        result.root / "anchors" / "promoter_controls_demo",
        result.root / "templates" / "pdual10_demo",
    ]
    if any(path.exists() for path in legacy_paths):
        typer.echo(
            "warning: legacy demo datasets from older ontology revisions are still present in this root; "
            "canonical curated inputs are mg1655_promoters and plasmids."
        )
