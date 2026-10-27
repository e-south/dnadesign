"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/opal_smoke.py

Exercise an installed OPAL wheel using two rounds of local candidate data.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import dnadesign.opal

_CLI = """
import importlib.abc
import sys
attempts = []
class RejectUSR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dnadesign.usr' or fullname.startswith('dnadesign.usr.'):
            attempts.append(fullname)
            raise ImportError('Local OPAL must not import USR: ' + fullname)
sys.meta_path.insert(0, RejectUSR())
from dnadesign.opal import main
try:
    main()
finally:
    assert not attempts, attempts
"""


def check(root: Path) -> None:
    for module in ("torch", "pymc", "scanpy", "ortools", "marimo", "dense_arrays"):
        assert find_spec(module) is None, module
    assert Path(dnadesign.opal.__file__).is_relative_to(sys.prefix)
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    example = Path(dnadesign.opal.__file__).parent / "campaigns/demo_rf_topn/configs/campaign.yaml"
    config = yaml.safe_load(example.read_text())
    config["ownership"] = {
        "owner_scope": "study_campaign",
        "study_id": "installation_check",
        "dataset_id": "candidates",
        "portable": False,
    }
    config["campaign"] = {"name": "Local table example", "slug": "local_table", "workdir": str(root)}
    config["data"]["location"] = {"kind": "local", "path": str(root / "records.parquet")}
    config["model"]["params"]["n_jobs"] = 1
    config.pop("plot_config", None)
    config_path = root / "configs/campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    sequences = ["ACGTACGT" + "".join("ACGT"[(i >> (2 * j)) & 3] for j in range(4)) for i in range(24)]
    features = [[float(i), float(i % 3), float(i % 5), float(i % 7)] for i in range(24)]
    pq.write_table(
        pa.table(
            {
                "id": [f"candidate_{i}" for i in range(24)],
                "sequence": sequences,
                "bio_type": ["dna"] * 24,
                "alphabet": ["dna_4"] * 24,
                "X": pa.array(features, type=pa.list_(pa.float64(), 4)),
            }
        ),
        root / "records.parquet",
    )

    def cli(*args: str) -> None:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _CLI, *args],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=60,
        )
        with (root / "commands.log").open("a") as stream:
            stream.write(f"opal {' '.join(args)}\n{result.stdout}{result.stderr}\n")
        assert result.returncode == 0, result.stdout + result.stderr

    cfg = ["-c", str(config_path)]
    cli("validate", *cfg)
    cli("init", *cfg)
    observed = [sequences[i] for i in (0, 4, 8, 12, 16, 20)]
    selected_counts = []
    for round_index in range(2):
        labels = root / f"labels-{round_index}.csv"
        with labels.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sequence", "y"])
            writer.writerows((sequence, 1 + 0.5 * sequences.index(sequence)) for sequence in observed)
        cli(
            "ingest-y",
            *cfg,
            "--round",
            str(round_index),
            "--csv",
            str(labels),
            "--unknown-sequences",
            "error",
            "--apply",
        )
        cli("run", *cfg, "--round", str(round_index))
        cli("verify-outputs", *cfg, "--view", "primary", "--round", "latest")
        selected = root / f"selected-{round_index}.csv"
        cli(
            "selection-set",
            "export",
            *cfg,
            "--view",
            "primary",
            "--round",
            "latest",
            "--out",
            str(selected),
            "--format",
            "csv",
        )
        with selected.open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows and {"id", "sequence", "selection_rank", "score"} <= rows[0].keys()
        assert all(row["sequence"] in sequences for row in rows)
        assert not set(observed).intersection(row["sequence"] for row in rows)
        observed = [row["sequence"] for row in rows]
        selected_counts.append(len(rows))
    (root / "verification.json").write_text(json.dumps({"rounds": 2, "selected_counts": selected_counts}))


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
