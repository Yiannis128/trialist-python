# Author: Yiannis Charalambous


import sys
from typing import TextIO

from IPython.display import display as display_widget
import ipywidgets as widgets


class TrialViewer:
    """Shows a progress report with the experiments."""

    def __init__(self, title: str = "") -> None:
        self._output: widgets.Output = widgets.Output()
        self._progress_bar: widgets.IntProgress = widgets.IntProgress(
            value=0, min=0, max=1
        )
        self._label: widgets.Label = widgets.Label(value=title)

        self._orig_stdout: TextIO

    def display(self) -> None:
        """Displays the widgets."""

        # Clear logging
        self._output.clear_output()

        # Attach stdout
        self._orig_stdout = sys.stdout
        sys.stdout = self._output

        display_widget(
            widgets.HBox(
                [
                    self._progress_bar,
                    self._label,
                ],
            ),
            self._output,
        )

    @property
    def label(self) -> str:
        """The text of the label."""
        return self._label.value

    @label.setter
    def label(self, value: str) -> None:
        self._label.value = value

    @property
    def progress(self) -> float:
        """Set the progress of the experiments"""
        return self._progress_bar.value

    @progress.setter
    def progress(self, value: float) -> None:
        self._progress_bar.value = value

    @property
    def max_count(self) -> float:
        """Set the max_count of the experiments"""
        return self._progress_bar.max

    @max_count.setter
    def max_count(self, value: float) -> None:
        self._progress_bar.max = value
