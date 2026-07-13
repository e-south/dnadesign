#!/usr/bin/env python3
"""Apply the canonical role-aware ChimeraX protein-DNA-RNA review style."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _commands(args: argparse.Namespace) -> list[str]:
    nucleic = args.nucleic_selection
    commands = [
        f"rename {args.model_selection} molecular_complex",
        f"name protein_role {args.protein_selection}",
        f"name dna_role {args.dna_selection}",
        f"name rna_role {args.rna_selection}",
        f"hide {args.protein_selection} atoms",
        f"cartoon {args.protein_selection}",
    ]
    if args.nucleic_display == "ladder":
        commands.append(f"nucleotides {nucleic} ladder")
    else:
        commands.append(f"nucleotides {nucleic} atoms")
    if args.nucleic_backbone_mode == "native":
        commands.extend(
            [
                f"cartoon {nucleic} suppressBackboneDisplay true",
                "cartoon style nucleic xsect oval width 1.35 thick 0.28",
                "cartoon tether nucleic shape cylinder sides 8 scale 0.65 opacity 1",
            ]
        )
    else:
        commands.append(f"hide {nucleic} cartoons")
    if args.nucleic_display == "connected-atoms":
        commands.extend(
            [
                f"show {nucleic} atoms",
                f"style {nucleic} stick",
                f"size {nucleic} stickRadius {args.nucleic_stick_radius:.2f}",
            ]
        )
    if args.nucleic_backbone_mode == "phosphate-ribbon":
        ribbon_specs = [
            (args.dna_phosphate_selection, args.dna_color, "#20", "dna_backbone"),
            *[
                (selection, args.rna_color, f"#{21 + index}", f"rna_backbone_{index + 1}")
                for index, selection in enumerate(args.rna_phosphate_selection)
            ],
        ]
        for selection, color, model_id, model_name in ribbon_specs:
            commands.extend(
                [
                    f"shape ribbon {selection} width 1.25 height 0.16 followBonds false "
                    f"color {color} modelId {model_id}",
                    f"rename {model_id} {model_name}",
                ]
            )
    commands.extend(
        [
            f"color {args.protein_selection} {args.protein_color} target c",
            f"color {args.dna_selection} {args.dna_color} target acf",
            f"color {args.rna_selection} {args.rna_color} target acf",
            f"surface {args.protein_selection}",
            f"color {args.protein_selection} {args.protein_color} target s",
            f"transparency {args.protein_selection} {args.surface_transparency} target s",
            f"view all pad {args.view_padding:.2f}",
        ]
    )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply role-aware nucleotide and protein-surface styles to an open ChimeraX complex."
    )
    endpoint = parser.add_mutually_exclusive_group(required=False)
    endpoint.add_argument("--port", type=int)
    endpoint.add_argument("--session-manifest", type=Path)
    parser.add_argument("--model-selection", default="#1")
    parser.add_argument("--protein-selection", required=True)
    parser.add_argument("--dna-selection", required=True)
    parser.add_argument("--rna-selection", required=True)
    parser.add_argument("--nucleic-selection", required=True)
    parser.add_argument(
        "--nucleic-display",
        choices=("ladder", "connected-atoms"),
        default="ladder",
    )
    parser.add_argument(
        "--nucleic-backbone-mode",
        choices=("native", "phosphate-ribbon"),
        default="native",
    )
    parser.add_argument("--dna-phosphate-selection")
    parser.add_argument("--rna-phosphate-selection", action="append", default=[])
    parser.add_argument("--protein-color", default="#E8E4DA")
    parser.add_argument("--dna-color", default="#B97700")
    parser.add_argument("--rna-color", default="#C84C5A")
    parser.add_argument("--surface-transparency", type=int, default=35)
    parser.add_argument("--nucleic-stick-radius", type=float, default=0.20)
    parser.add_argument("--view-padding", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.surface_transparency <= 100:
        parser.error("--surface-transparency must be from 0 to 100")
    if args.nucleic_stick_radius <= 0:
        parser.error("--nucleic-stick-radius must be positive")
    if args.view_padding < 0:
        parser.error("--view-padding must be non-negative")
    if args.nucleic_backbone_mode == "phosphate-ribbon":
        if args.nucleic_display != "connected-atoms":
            parser.error("phosphate-ribbon mode requires --nucleic-display connected-atoms")
        if not args.dna_phosphate_selection or not args.rna_phosphate_selection:
            parser.error(
                "phosphate-ribbon mode requires --dna-phosphate-selection and at least one --rna-phosphate-selection"
            )
    if not args.dry_run and args.port is None and args.session_manifest is None:
        parser.error("provide --port or --session-manifest unless --dry-run is used")

    commands = _commands(args)
    if args.dry_run:
        json.dump({"schema_id": "chimerax_role_aware_complex_style_v1", "commands": commands}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    sender = Path(__file__).with_name("chimerax-send-command.py")
    for command in commands:
        sender_args = [sys.executable, str(sender)]
        if args.session_manifest is not None:
            sender_args.extend(["--session-manifest", str(args.session_manifest)])
        else:
            sender_args.extend(["--port", str(args.port)])
        sender_args.extend(["--command", command])
        subprocess.run(sender_args, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
