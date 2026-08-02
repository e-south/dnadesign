"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/banners/catalog.py

Defines the README banner inventory and plain capability labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BannerSpec:
    path: str
    name: str
    capability: str
    description: str
    glyph: str


REPOSITORY_BANNER_PATH = "assets/dnadesign-banner.svg"


BANNERS = (
    BannerSpec(
        "src/dnadesign/aligner/assets/aligner-banner.svg",
        "aligner",
        "ALIGN SEQUENCES",
        "Align nucleotide pairs and protein sequence sets.",
        "align",
    ),
    BannerSpec(
        "src/dnadesign/baserender/assets/baserender-banner.svg",
        "baserender",
        "DRAW SEQUENCE MAPS",
        "Draw annotated sequence diagrams from saved job files.",
        "render",
    ),
    BannerSpec(
        "src/dnadesign/billboard/assets/billboard-banner.svg",
        "billboard",
        "MEASURE DIVERSITY",
        "Measure binding-site diversity in generated libraries.",
        "diversity",
    ),
    BannerSpec(
        "src/dnadesign/cluster/assets/cluster-banner.svg",
        "cluster",
        "GROUP FEATURES",
        "Group and visualize rows in numerical feature tables.",
        "cluster",
    ),
    BannerSpec(
        "src/dnadesign/construct/docs/assets/construct-banner.svg",
        "construct",
        "ASSEMBLE CONSTRUCTS",
        "Place named DNA parts into declared sequence contexts.",
        "construct",
    ),
    BannerSpec(
        "src/dnadesign/contracts/assets/contracts-banner.svg",
        "contracts",
        "SHARE DATA FORMATS",
        "Define versioned data formats shared between tools.",
        "contracts",
    ),
    BannerSpec(
        "src/dnadesign/cruncher/assets/cruncher-banner.svg",
        "cruncher",
        "DESIGN TO CONSTRAINTS",
        "Find sequences and assembly plans that satisfy stated constraints.",
        "optimize",
    ),
    BannerSpec(
        "src/dnadesign/densegen/assets/densegen-banner.svg",
        "densegen",
        "GENERATE LIBRARIES",
        "Generate reproducible DNA libraries from workspace inputs.",
        "generate",
    ),
    BannerSpec(
        "src/dnadesign/devtools/tests/support/assets/testsupport-banner.svg",
        "testsupport",
        "SHARE TEST FIXTURES",
        "Share test-only fixtures across repository tools.",
        "test",
    ),
    BannerSpec(
        "src/dnadesign/folding/assets/folding-banner.svg",
        "folding",
        "DRAW RNA STRUCTURES",
        "Predict and draw RNA secondary structures from sequence records.",
        "fold",
    ),
    BannerSpec(
        "src/dnadesign/infer/assets/infer-banner.svg",
        "infer",
        "ADD MODEL RESULTS",
        "Run sequence models and add their results to datasets.",
        "infer",
    ),
    BannerSpec(
        "src/dnadesign/latentdna/docs/assets/latentdna-banner.svg",
        "latentdna",
        "COMPARE SEQUENCE DATA",
        "Compare numerical sequence features and save review-ready results.",
        "latent",
    ),
    BannerSpec(
        "src/dnadesign/libshuffle/assets/libshuffle-banner.svg",
        "libshuffle",
        "SELECT DIVERSE SETS",
        "Select representative subsets from dense sequence libraries.",
        "shuffle",
    ),
    BannerSpec(
        "src/dnadesign/nmf/assets/nmf-banner.svg",
        "nmf",
        "FIND MOTIF PATTERNS",
        "Find recurring combinations in motif tables.",
        "factor",
    ),
    BannerSpec(
        "src/dnadesign/notify/assets/notify-banner.svg",
        "notify",
        "SEND RUN UPDATES",
        "Send notifications for local and scheduled jobs.",
        "notify",
    ),
    BannerSpec(
        "src/dnadesign/opal/assets/opal-banner.svg",
        "opal",
        "SELECT NEXT DESIGNS",
        "Choose the next designs from measured sequence results.",
        "select",
    ),
    BannerSpec(
        "src/dnadesign/ops/assets/ops-banner.svg",
        "ops",
        "FIND AND RUN JOBS",
        "Find, run, and inspect jobs owned by repository tools.",
        "route",
    ),
    BannerSpec(
        "src/dnadesign/permuter/assets/permuter-banner.svg",
        "permuter",
        "MAKE AND SCORE VARIANTS",
        "Generate sequence variants and score them with chosen tools.",
        "permute",
    ),
    BannerSpec(
        "src/dnadesign/studies/assets/studies-banner.svg",
        "studies",
        "ORGANIZE STUDIES",
        "Keep study methods, records, and decisions together.",
        "study",
    ),
    BannerSpec(
        "src/dnadesign/tfkdanalysis/assets/tfkdanalysis-banner.svg",
        "tfkdanalysis",
        "SUMMARIZE KNOCKDOWNS",
        "Summarize transcription-factor knockdown responses.",
        "knockdown",
    ),
    BannerSpec(
        "src/dnadesign/thread/assets/thread-banner.svg",
        "thread",
        "PREPARE PROTEIN DESIGNS",
        "Prepare fixed-backbone design requests and fold checks.",
        "thread",
    ),
    BannerSpec(
        "src/dnadesign/junction/assets/junction-banner.svg",
        "junction",
        "TARGETS / OLIGOS / CHECKS",
        "Compile exact DNA targets into checked three-way-junction oligo plans.",
        "junction",
    ),
    BannerSpec(
        "src/dnadesign/usr/assets/usr-banner.svg",
        "usr",
        "MANAGE SEQUENCE TABLES",
        "Store, check, and sync tables of sequence records.",
        "records",
    ),
)
