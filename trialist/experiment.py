# Author: Yiannis Charalambous

from pathlib import Path
from typing import Any, Callable, NamedTuple
from math import prod
from itertools import product
import joblib
from loguru import logger


class Experiment(NamedTuple):
    """Represents a single experiment.

    Args:
        params: dictionary of <string, int> pairs where the string indicates the
            parameter, and the int is the index of the current experiment.
        idx: the index of this experiment. Don't use this in the key-gen function
            if the order is due to change. Use all the other parameters of the
            experiment that are constant to this experiment.
        max_count: the amount of total experiments to run."""

    params: dict[str, int]
    idx: int
    max_count: int


class ExperimentResult(NamedTuple):
    """Results for the experiments."""

    experiment: Experiment
    result: Any


class Checkpoint:
    """Checkpoint system for experimental trials. Allows for saving/restoring of
    experiment results."""

    def __init__(
        self,
        checkpoint_dir: Path,
        clear_names: bool = True,
        delete_invalid: bool = True,
        validate_fn: Callable[[Any], bool] = lambda *_: True,
        logger2: Any = None,
    ) -> None:
        self._checkpoint_dir: Path = checkpoint_dir
        self._clear_names: bool = clear_names
        self._logger = logger2 if logger2 else logger.bind(module="Trialist")
        self._delete_invalid: bool = delete_invalid
        self._validate_fn: Callable[[Any], bool] = validate_fn

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        """Return the name of the checkpoint directory."""
        return self._checkpoint_dir

    def check(self, key: str) -> Any:
        """Checks if a checkpoint exists, if it does, it returns the cache content."""
        checkpoint_file = Path(self._checkpoint_dir / key)
        if checkpoint_file.exists():
            res = joblib.load(checkpoint_file)

            if self._validate_fn(res):
                self._logger.info(f"Found in cache: {key}... Restoring...")
                return res

            if self._delete_invalid:
                self._logger.info(f"Invalid checkpoint: {key}... Deleting...")
                checkpoint_file.unlink()
            else:
                self._logger.info(f"Invalid checkpoint: {key}...")

            return None

    def save(self, key: str, result: Any) -> None:
        """Saves a checkpoint overwriting the old one if it exists."""
        checkpoint_file = Path(self._checkpoint_dir / key)
        joblib.dump(result, checkpoint_file)
        self._logger.info(f"Checkpoint saved: {checkpoint_file}")

    def discard(self, key: str) -> None:
        """Deletes a checkpoint file if it exists."""
        checkpoint_file = Path(self._checkpoint_dir / key)
        if checkpoint_file.exists():
            checkpoint_file.unlink(True)
            self._logger.info(f"Checkpoint discarded: {checkpoint_file}")


class Trials:
    """Class for running experiments in a loop."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        exp_fn: Callable[[Experiment], Any],
        key_gen: Callable[[Experiment], str],
    ) -> None:
        self._checkpoint: Checkpoint = checkpoint
        self._epoch_fn: Callable[[Experiment], Any] = exp_fn
        self._key_gen: Callable[[Experiment], str] = key_gen

    @property
    def checkpoint_names(self) -> list[str]:
        """Returns a list of checkpoint names."""
        return [
            f.name for f in self._checkpoint.checkpoint_dir.iterdir() if f.is_file()
        ]

    def run(
        self,
        counts_matrix: list[tuple[str, int]],
    ) -> list[ExperimentResult]:
        """Runs the experiments and returns the results.

        Args:
            counts: list of tuples of param names and count of tests to run for
                each param."""

        results: list[ExperimentResult] = []
        # Unzip into two sequences: names and their counts
        names_tuple, counts_tuple = zip(*counts_matrix)
        names: list[str] = list(names_tuple)
        counts: list[int] = list(counts_tuple)
        del names_tuple, counts_tuple

        max_count: int = prod(counts)

        # Make a range(1…count) for each entry
        ranges: list[range] = [range(0, c) for c in counts]

        # product(*) will iterate over every possible combination
        # combo is a tuple that contains a configuration of the counts_matrix
        for exp_idx, combo in enumerate(product(*ranges)):
            idx_map = dict(zip(names, combo))
            exp = Experiment(
                params=idx_map,
                idx=exp_idx,
                max_count=max_count,
            )
            key: str = self._key_gen(exp)

            # Checkpoint Check
            result: Any = self._checkpoint.check(key)
            if result is None:
                result = self._epoch_fn(exp)
                # Checkpoint Save
                self._checkpoint.save(key, result)

            # Store experiment
            results.append(
                ExperimentResult(
                    result=result,
                    experiment=exp,
                )
            )
        return results

    def clear_checkpoints(self) -> None:
        """Deletes all checkpoint files."""
        for f in self._checkpoint.checkpoint_dir.iterdir():
            if f.is_file():
                f.unlink()
