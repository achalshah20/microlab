"""Checkpoint durability: the layer M4's ~50-session chain depends on.

These tests simulate the failure modes a preemptible box actually produces —
a write interrupted partway, a truncated file, a missing commit marker — rather
than only the happy path.
"""

from __future__ import annotations

import json

import pytest
import torch

from microlab.train.checkpoint import (
    CHECKPOINT_VERSION,
    checkpoint_paths,
    list_checkpoints,
    load_checkpoint,
    load_latest,
    prune_checkpoints,
    save_checkpoint,
)


def _save(directory, step, value=1.0, keep=10):
    return save_checkpoint(
        directory,
        step=step,
        payload={"model": {"w": torch.tensor([value])}, "marker": step},
        run_id="test",
        session=0,
        tokens_seen=step * 100,
        keep_last_n=keep,
    )


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        _save(tmp_path, 10, value=3.5)
        payload, path = load_latest(tmp_path)
        assert payload["step"] == 10
        assert payload["version"] == CHECKPOINT_VERSION
        torch.testing.assert_close(payload["model"]["w"], torch.tensor([3.5]))
        assert path.name == "step_00000010.pt"

    def test_none_when_empty(self, tmp_path):
        assert load_latest(tmp_path) is None
        assert load_latest(tmp_path / "does_not_exist") is None

    def test_latest_wins(self, tmp_path):
        for step in (5, 100, 20):
            _save(tmp_path, step)
        payload, _ = load_latest(tmp_path)
        assert payload["step"] == 100

    def test_meta_records_provenance(self, tmp_path):
        _save(tmp_path, 7)
        _, meta_path = checkpoint_paths(tmp_path, 7)
        meta = json.loads(meta_path.read_text())
        assert meta["step"] == 7
        assert meta["tokens_seen"] == 700
        assert meta["run_id"] == "test"
        assert len(meta["sha256"]) == 64
        assert meta["torch_version"] == torch.__version__

    def test_hash_verification_passes_on_good_file(self, tmp_path):
        path = _save(tmp_path, 3)
        assert load_checkpoint(path, verify_hash=True)["step"] == 3


class TestInterruptedWrites:
    def test_payload_without_marker_is_ignored(self, tmp_path):
        """A session killed during torch.save leaves exactly this state."""
        _save(tmp_path, 10)
        _save(tmp_path, 20)
        _, meta20 = checkpoint_paths(tmp_path, 20)
        meta20.unlink()  # simulate: payload written, marker never was

        assert [s for s, _, _ in list_checkpoints(tmp_path)] == [10]
        payload, _ = load_latest(tmp_path)
        assert payload["step"] == 10

    def test_corrupt_payload_falls_back_to_previous(self, tmp_path):
        """One bad checkpoint costs one interval, not the run."""
        _save(tmp_path, 10)
        newest = _save(tmp_path, 20)
        newest.write_bytes(b"this is not a torch file")

        payload, path = load_latest(tmp_path)
        assert payload["step"] == 10
        assert path.name == "step_00000010.pt"

    def test_truncated_payload_falls_back(self, tmp_path):
        _save(tmp_path, 10)
        newest = _save(tmp_path, 20)
        data = newest.read_bytes()
        newest.write_bytes(data[: len(data) // 2])

        payload, _ = load_latest(tmp_path)
        assert payload["step"] == 10

    def test_all_corrupt_raises_rather_than_silently_restarting(self, tmp_path):
        """Losing every checkpoint must be loud.

        Returning None here would look like "fresh run" and silently restart a
        multi-week run from step 0.
        """
        path = _save(tmp_path, 10)
        path.write_bytes(b"garbage")
        with pytest.raises(RuntimeError, match="failed to load"):
            load_latest(tmp_path)

    def test_hash_mismatch_detected(self, tmp_path):
        path = _save(tmp_path, 10)
        _, meta_path = checkpoint_paths(tmp_path, 10)
        meta = json.loads(meta_path.read_text())
        meta["sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="hash check"):
            load_checkpoint(path, verify_hash=True)

    def test_no_temp_files_survive_a_successful_save(self, tmp_path):
        _save(tmp_path, 10)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_version_mismatch_rejected(self, tmp_path):
        path = _save(tmp_path, 10)
        payload = torch.load(path, weights_only=False)
        payload["version"] = 999
        torch.save(payload, path)
        with pytest.raises(RuntimeError, match="failed to load"):
            load_latest(tmp_path)


class TestPruning:
    def test_keeps_only_newest_n(self, tmp_path):
        for step in range(1, 6):
            _save(tmp_path, step * 10, keep=2)
        assert [s for s, _, _ in list_checkpoints(tmp_path)] == [50, 40]
        assert len(list(tmp_path.glob("*.pt"))) == 2

    def test_prune_removes_marker_and_payload_together(self, tmp_path):
        for step in (1, 2, 3):
            _save(tmp_path, step, keep=10)
        prune_checkpoints(tmp_path, keep_last_n=1)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "step_00000003.meta.json",
            "step_00000003.pt",
        ]

    def test_keep_zero_is_a_noop(self, tmp_path):
        _save(tmp_path, 1, keep=10)
        assert prune_checkpoints(tmp_path, 0) == []
        assert len(list_checkpoints(tmp_path)) == 1
