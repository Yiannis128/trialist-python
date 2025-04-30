# Author: Yiannis Charalambous


from pathlib import Path
import pytest
import joblib
from loguru import logger

from trialist import Checkpoint


@pytest.fixture(autouse=True)
def patch_logger():
    """Remove the logger since not needed."""
    logger.remove()


def test_checkpoint_dir_property(tmp_path):
    """Ensure that the checkpoint_dir property returns the directory passed to __init__."""
    cp = Checkpoint(checkpoint_dir=tmp_path, clear_names=False)
    assert cp.checkpoint_dir == tmp_path


def test_save_creates_file_and_can_load(tmp_path):
    """Verify that save() writes a file and that joblib.load can read back the same data."""
    cp = Checkpoint(checkpoint_dir=tmp_path)
    key = "foo"
    # Example data
    result = {"a": 123, "log_file": str(tmp_path / "dummy.log")}
    cp.save(key, result)

    # file should now exist
    checkpoint_file = tmp_path / key
    assert checkpoint_file.exists()

    # if we load it directly via joblib we get the same dict
    loaded = joblib.load(checkpoint_file)
    assert loaded == result


def test_check_returns_saved_result_if_valid(tmp_path):
    """Check() should return the checkpoint dict when the log_file exists."""
    cp = Checkpoint(checkpoint_dir=tmp_path)
    key = "valid.chk"
    result = {"foo": "bar", "log_file": str(tmp_path / "log.txt")}
    joblib.dump(result, tmp_path / key)

    returned = cp.check(key)
    assert returned == result


def test_check_returns_none_if_no_checkpoint(tmp_path):
    """Check() should return None when the checkpoint file does not exist."""
    cp = Checkpoint(checkpoint_dir=tmp_path)
    returned = cp.check("does_not_exist.chk")
    assert returned is None


def test_discard_removes_existing_file(tmp_path):
    """Discard() should delete an existing checkpoint file."""
    cp = Checkpoint(checkpoint_dir=tmp_path)
    key = "to_discard.chk"
    f = tmp_path / key
    f.write_text("x")
    assert f.exists()

    cp.discard(key)
    assert not f.exists()


def test_discard_on_nonexistent(tmp_path):
    """Discard() should not raise an error when the checkpoint file is missing."""
    cp = Checkpoint(checkpoint_dir=tmp_path)
    key = "no_file.chk"
    # should not raise
    cp.discard(key)
    assert not (tmp_path / key).exists()
