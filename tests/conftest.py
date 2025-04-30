# Author: Yiannis Charalambous


from shutil import rmtree
import tempfile
from pathlib import Path
import pytest

from trialist import Experiment

# Keep track of all temp dirs we hand out
_temp_dirs: list[Path] = []


@pytest.fixture
def temp_checkpoint_dir():
    """Provide a temporary directory for checkpoints, and register it for later cleanup."""
    tmpdir: Path = Path(tempfile.mkdtemp())
    _temp_dirs.append(tmpdir)
    yield Path(tmpdir)
    # Note: we _don’t_ delete here, so pytest_sessionfinish can handle it all in one place.


@pytest.fixture
def mock_epoch_fn():
    """Return a exp function that always returns a valid result dict."""

    def _fn(exp: Experiment):
        return {"success": exp.idx % 2 == 0, "result": exp.idx, **exp.params}

    return _fn


@pytest.fixture
def key_gen():
    """Return a key-generation function matching our Trials.save/load filenames."""
    return lambda exp: f"{exp.idx}"


def pytest_sessionfinish(session, exitstatus):
    """
    After all tests have run, delete every temporary directory we created.
    This ensures no leftover files linger on disk.
    """
    _ = session, exitstatus
    for tmpdir in _temp_dirs:
        rmtree(tmpdir)
