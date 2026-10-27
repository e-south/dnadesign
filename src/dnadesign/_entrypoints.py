"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_entrypoints.py

Lazy installed command launchers with optional-runtime diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from importlib import import_module

# Only diagnose imports supplied by the recommended installation profile.
# Unknown failures, including command/plugin defects, retain their traceback.
_CLI_IMPORTS = frozenset({"pandas", "pyarrow", "pydantic", "yaml", "rich", "typer"})
_PROFILE_IMPORTS = {
    "usr": _CLI_IMPORTS | {"Bio", "duckdb", "psutil"},
    "permuter": _CLI_IMPORTS | {"matplotlib", "seaborn", "psutil"},
    "opal": _CLI_IMPORTS | {"polars", "sklearn", "scipy", "joblib", "typing_extensions", "matplotlib", "openpyxl"},
    "full": _CLI_IMPORTS
    | {
        "anyio",
        "arviz",
        "Bio",
        "click",
        "igraph",
        "leidenalg",
        "logomaker",
        "markdown_it",
        "matplotlib",
        "openpyxl",
        "ortools",
        "PIL",
        "pygments",
        "Levenshtein",
        "scanpy",
        "sklearn",
        "scipy",
        "seaborn",
        "tqdm",
        "umap",
        "duckdb",
        "xlrd",
        "torch",
        "dense_arrays",
        "polars",
        "altair",
        "vegafusion",
        "vl_convert",
        "pysam",
        "netCDF4",
        "certifi",
        "keyring",
        "fastparquet",
        "filelock",
        "marimo",
        "RNA",
        "zstd",
        "py3Dmol",
        "plotly",
        "psutil",
        "typing_extensions",  # Required by the profile's Pydantic dependency.
    },
    "infer-evo2": {"evo2", "flash_attn", "transformer_engine"},
}
_PROFILE_IMPORTS["infer"] = _PROFILE_IMPORTS["usr"] | {"tqdm"}


def _run(target: str, extra: str):
    try:
        module, attribute = target.split(":")
        return getattr(import_module(module), attribute)()
    except ModuleNotFoundError as exc:
        if not exc.name or exc.name.startswith("dnadesign") or "." in exc.name:
            raise
        if extra in {"infer", "permuter", "full"}:
            if exc.name in {"evo2", "flash_attn", "transformer_engine"}:
                extra = "infer-evo2"
            elif exc.name == "torch":
                extra = "full"
        if exc.name.split(".")[0] not in _PROFILE_IMPORTS.get(extra, ()):
            raise
        raise SystemExit(
            f"This command requires dnadesign-tools[{extra}] (missing {exc.name}). "
            f"Install with pip install 'dnadesign-tools[{extra}]', or in a checkout "
            f"run uv sync --locked --no-default-groups --extra {extra}."
        ) from None


def usr():
    return _run("dnadesign.usr:main", "usr")


def usr_sync_audit_drill():
    return _run("dnadesign.usr.ops.sync_audit_drill:main", "usr")


def opal():
    return _run("dnadesign.opal:main", "opal")


def dense():
    return _run("dnadesign.densegen:app", "full")


def notify():
    return _run("dnadesign.notify.cli:app", "full")


def infer():
    return _run("dnadesign.infer.cli:app", "infer")


def construct():
    return _run("dnadesign.construct:main", "full")


def folding():
    return _run("dnadesign.folding.cli:main", "full")


def cluster():
    return _run("dnadesign.cluster.cli:main", "full")


def latentdna():
    return _run("dnadesign.latentdna.src:main", "full")


def permuter():
    return _run("dnadesign.permuter:main", "permuter")


def baserender():
    return _run("dnadesign.baserender:app", "full")


def cruncher():
    return _run("dnadesign.cruncher.cli.app:app", "full")


def junction():
    return _run("dnadesign.junction.cli:main", "full")


def ops():
    return _run("dnadesign.ops.cli:main", "full")
