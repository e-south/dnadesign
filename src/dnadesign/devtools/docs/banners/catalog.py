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
        "RENDER SEQUENCES",
        "Render sequence records from explicit visual contracts.",
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
        "EXPLORE FEATURES",
        "Explore feature tables through clusters and projections.",
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
        "DEFINE HANDOFFS",
        "Define neutral, versioned artifacts shared between tools.",
        "contracts",
    ),
    BannerSpec(
        "src/dnadesign/cruncher/assets/cruncher-banner.svg",
        "cruncher",
        "OPTIMIZE SEQUENCES",
        "Solve explicit sequence-design and assembly workflows.",
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
        "PREDICT STRUCTURE",
        "Predict and render secondary structure for sequence artifacts.",
        "fold",
    ),
    BannerSpec(
        "src/dnadesign/infer/assets/infer-banner.svg",
        "infer",
        "RUN SEQUENCE MODELS",
        "Run sequence models and publish namespaced results.",
        "infer",
    ),
    BannerSpec(
        "src/dnadesign/latentdna/docs/assets/latentdna-banner.svg",
        "latentdna",
        "COMPARE REPRESENTATIONS",
        "Compare learned sequence representations in workspaces.",
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
        "FIND MOTIF PROGRAMS",
        "Factor motif tables into recurring programs.",
        "factor",
    ),
    BannerSpec(
        "src/dnadesign/notify/assets/notify-banner.svg",
        "notify",
        "WATCH DATA EVENTS",
        "Watch sequence-record events and send notifications.",
        "notify",
    ),
    BannerSpec(
        "src/dnadesign/opal/assets/opal-banner.svg",
        "opal",
        "SELECT NEXT DESIGNS",
        "Run contract-driven active-learning campaigns.",
        "select",
    ),
    BannerSpec(
        "src/dnadesign/ops/assets/ops-banner.svg",
        "ops",
        "RUN SHARED WORKFLOWS",
        "Discover, inspect, and run tool-owned workflows.",
        "route",
    ),
    BannerSpec(
        "src/dnadesign/permuter/assets/permuter-banner.svg",
        "permuter",
        "GENERATE VARIANTS",
        "Generate and score sequence variants through explicit evaluators.",
        "permute",
    ),
    BannerSpec(
        "src/dnadesign/studies/assets/studies-banner.svg",
        "studies",
        "OWN STUDY LOGIC",
        "Keep study-specific code with its checked-in records.",
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
        "src/dnadesign/usr/assets/usr-banner.svg",
        "usr",
        "STORE SEQUENCE RECORDS",
        "Store sequence records, overlays, and mutation events.",
        "records",
    ),
)
