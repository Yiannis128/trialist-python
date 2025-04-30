# Author: Yiannis Charalambous

from pathlib import Path
import joblib
import pytest

from trialist import Experiment, Checkpoint, Trials


def test_run_and_cache_behavior(temp_checkpoint_dir, mock_epoch_fn, key_gen):
    """
    First run should generate results and create checkpoint files;
    second run should load from cache.
    """
    checkpoint = Checkpoint(temp_checkpoint_dir)
    trials = Trials(checkpoint, mock_epoch_fn, key_gen)

    # First run: creates two checkpoints
    first = trials.run([("param", 2)])
    expected = [
        {"result": 0, "success": True, "param": 0},
        {"result": 1, "success": False, "param": 1},
    ]
    assert first[0].result == expected[0]
    assert first[1].result == expected[1]

    # Second Run: Ensure that the pkl file created is actually loaded.
    files = sorted(p.name for p in temp_checkpoint_dir.iterdir())

    # Modify first checkpoint on disk.
    modified = {"result": 3, "success": True}
    joblib.dump(modified, temp_checkpoint_dir / files[0])

    # Second run: should pick up modified cache for idx=0 and cached for idx=1
    second = trials.run([("param", 2)])
    assert second[0].result["result"] == 3
    assert second[1].result["result"] == 1


def test_checkpoint_names_and_clear(temp_checkpoint_dir, mock_epoch_fn, key_gen):
    """
    checkpoint_names should list all .pkl files;
    clear_checkpoints should remove them all.
    """
    checkpoint = Checkpoint(temp_checkpoint_dir)
    trials = Trials(checkpoint, mock_epoch_fn, key_gen)

    trials.run([("a", 2), ("b", 1)])
    names = sorted(trials.checkpoint_names)
    assert names == ["0", "1"]

    trials.clear_checkpoints()
    assert trials.checkpoint_names == []
    assert not list(temp_checkpoint_dir.iterdir())
