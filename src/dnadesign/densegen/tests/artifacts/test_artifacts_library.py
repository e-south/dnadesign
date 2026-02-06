from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.core.artifacts.library import load_library_artifact, write_library_artifact


def test_write_library_artifact(tmp_path: Path) -> None:
    builds = [
        {
            "created_at": "2026-01-20T00:00:00+00:00",
            "input_name": "demo",
            "plan_name": "plan",
            "library_index": 1,
            "library_id": "libhash",
            "library_hash": "libhash",
            "pool_strategy": "subsample",
            "library_sampling_strategy": "tf_balanced",
            "library_size": 2,
            "achieved_length": 18,
            "relaxed_cap": False,
            "final_cap": None,
            "iterative_max_libraries": 0,
            "iterative_min_new_solutions": 0,
            "required_regulators_selected": None,
        }
    ]
    members = [
        {
            "library_id": "libhash",
            "library_hash": "libhash",
            "library_index": 1,
            "input_name": "demo",
            "plan_name": "plan",
            "position": 0,
            "tf": "TF1",
            "tfbs": "AAAA",
            "tfbs_id": "id1",
            "motif_id": "motif1",
            "site_id": None,
            "source": "src",
        }
    ]
    artifact = write_library_artifact(
        out_dir=tmp_path,
        builds=builds,
        members=members,
        cfg_path=Path("config.yaml"),
        run_id="demo",
        run_root=tmp_path,
        overwrite=True,
    )

    assert artifact.manifest_path.exists()
    assert artifact.builds_path.exists()
    assert artifact.members_path.exists()

    loaded = load_library_artifact(tmp_path)
    assert loaded.builds_path.name == artifact.builds_path.name
    assert loaded.members_path.name == artifact.members_path.name
