# Trialist

[![PyPI - Version](https://img.shields.io/pypi/v/trialist.svg)](https://pypi.org/project/trialist)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trialist.svg)](https://pypi.org/project/trialist)

-----

## Table of Contents

- [Installation](#installation)
- [Example](#example)
- [License](#license)

## Installation

```console
pip install trialist
```

## Example

The following example will run the experiments once, then the second time will
automatically load the results from cache.

```py
from pathlib import Path
from typing import Any
from time import sleep
from trialist import Trials, Experiment, Checkpoint


def experiment(exp: Experiment) -> Any:
    print(f"Starting expensive calculation {exp.params} #{exp.idx}/{exp.max_count}")
    sleep(1)
    return f"Result {exp.idx}"


trial = Trials(
    checkpoint=Checkpoint(checkpoint_dir=Path("./results")),
    epoch_fn=experiment,
    key_gen=lambda exp: str(exp.idx),
)

results = trial.run([("ai_model", 2), ("epochs", 5)])

print(results)
```

You can customize every part of the system:

* Key generation function: Determines if the cache for that expeirment exists.
* Validation function: Determines if the experiment is valid. By default, all
values are loaded without checks.

## License

See the LICENSE file.
