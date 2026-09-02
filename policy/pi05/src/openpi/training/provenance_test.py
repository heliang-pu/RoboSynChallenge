import dataclasses
import hashlib
import json

import numpy as np

import openpi.training.provenance as provenance


@dataclasses.dataclass(frozen=True)
class _ModelConfig:
    action_horizon: int = 50


@dataclasses.dataclass(frozen=True)
class _TrainConfig:
    model: _ModelConfig = dataclasses.field(default_factory=_ModelConfig)
    batch_size: int = 64

    @property
    def assets_dirs(self):
        return "/tmp/assets"

    @property
    def checkpoint_dir(self):
        return "/tmp/checkpoints"


@dataclasses.dataclass(frozen=True)
class _DataConfig:
    repo_id: str
    asset_id: str
    norm_stats: dict
    action_sequence_keys: tuple[str, ...] = ("action",)
    prompt_from_task: bool = True
    rlds_data_dir: str | None = None
    datasets: tuple = ()


def test_write_provenance_hashes_local_metadata(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text('{"total_episodes": 2}\n')
    data_config = _DataConfig(
        repo_id=str(dataset),
        asset_id="test/data",
        norm_stats={"state": {"mean": np.array([1.0, 2.0])}},
    )
    monkeypatch.setattr(provenance, "_launch_command_text", lambda: "command=test\n")
    monkeypatch.setattr(provenance, "_git_commit_text", lambda: "commit=test\n")

    output = provenance.write_provenance(tmp_path / "output", _TrainConfig(), data_config, checkpoint_step=12)

    assert {path.name for path in output.iterdir()} == {
        "train_config.json",
        "launch_command.txt",
        "git_commit.txt",
        "dataset_fingerprint.json",
    }
    train_config = json.loads((output / "train_config.json").read_text())
    assert train_config["checkpoint_step"] == 12
    assert train_config["config"]["model"]["action_horizon"] == 50

    dataset_info = json.loads((output / "dataset_fingerprint.json").read_text())
    assert dataset_info["metadata"]["file_count"] == 1
    assert dataset_info["metadata"]["total_bytes"] == len('{"total_episodes": 2}\n')
    assert dataset_info["norm_stats_sha256"] is not None
    assert (
        dataset_info["fingerprint_sha256"]
        == hashlib.sha256(
            json.dumps(
                {
                    key: dataset_info[key]
                    for key in (
                        "repo_id",
                        "asset_id",
                        "action_sequence_keys",
                        "prompt_from_task",
                        "rlds_data_dir",
                        "rlds_datasets",
                        "metadata_sha256",
                        "norm_stats_sha256",
                    )
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )

    (dataset / "meta" / "info.json").write_text('{"total_episodes": 3}\n')
    changed_output = provenance.write_provenance(
        tmp_path / "changed_output", _TrainConfig(), data_config, checkpoint_step=12
    )
    changed_dataset_info = json.loads((changed_output / "dataset_fingerprint.json").read_text())
    assert changed_dataset_info["fingerprint_sha256"] != dataset_info["fingerprint_sha256"]


def test_preserve_existing_writes_resume_history(tmp_path, monkeypatch):
    data_config = _DataConfig(repo_id="remote/repo", asset_id="remote/repo", norm_stats={})
    monkeypatch.setattr(provenance, "_launch_command_text", lambda: "command=initial\n")
    monkeypatch.setattr(provenance, "_git_commit_text", lambda: "commit=test\n")
    root = provenance.write_provenance(tmp_path / "output", _TrainConfig(), data_config)

    monkeypatch.setattr(provenance, "_launch_command_text", lambda: "command=resume\n")
    resumed = provenance.write_provenance(root, _TrainConfig(), data_config, preserve_existing=True)

    assert resumed.parent.parent == root
    assert (root / "launch_command.txt").read_text() == "command=initial\n"
    assert (resumed / "launch_command.txt").read_text() == "command=resume\n"
