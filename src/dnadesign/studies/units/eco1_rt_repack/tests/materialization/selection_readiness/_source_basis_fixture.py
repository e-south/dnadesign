"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_source_basis_fixture.py

Manual mask-authority source-basis fixture for selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml


def write_manual_mask_authority_source_basis(repo_root: Path) -> None:
    path = repo_root / "docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "source_basis": [
                    {
                        "id": "tao_et_al_2026_functional_residue_preservation",
                        "role": "method_prior",
                        "source_ref": "doi:10.1038/s41587-026-03149-6",
                    },
                    {
                        "id": "simon_et_al_2019_retron_rt_motif_grammar",
                        "role": "motif_annotation_prior",
                        "source_ref": "doi:10.1093/nar/gkz865",
                    },
                    {
                        "id": "wang_et_al_2022_ec86_cryoem_structure_priors",
                        "role": "ec86_structure_mask_prior",
                        "source_ref": "doi:10.1038/s41564-022-01197-7",
                    },
                    {
                        "id": "inouye_et_al_1999_ec86_primer_template_recognition",
                        "role": "c_terminal_specificity_review_prior",
                        "source_ref": "doi:10.1074/jbc.274.44.31236",
                    },
                    {
                        "id": "inouye_et_al_2004_ec86_thumb_primer_rna_binding",
                        "role": "c_terminal_specificity_review_prior",
                        "source_ref": "doi:10.1074/jbc.M408462200",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
