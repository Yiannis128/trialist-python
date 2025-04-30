# Trialist

[![PyPI - Version](https://img.shields.io/pypi/v/trialist.svg)](https://pypi.org/project/trialist)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trialist.svg)](https://pypi.org/project/trialist)
[![Build, Test, and Upload Artifact](https://github.com/Yiannis128/trialist-python/actions/workflows/workflow.yaml/badge.svg?branch=master)](https://github.com/Yiannis128/trialist-python/actions/workflows/workflow.yaml)

-----

## Table of Contents

- [Introduction](#introduction)
    - [Example](#example)
- [Installation](#installation)
- [License](#license)

## Introduction

Trialist is a library that manages and runs your experiments for you care-free.
It manages the experimental loop, logs, and sets checkpoints as well. If your
experiments are restarted, it will reload from the checkpoints set.

You can customize every part of the system:

* Key generation function: Determines how a sample is stored and found.
* Validation function: Determines if the experiment is valid. By default, all
experimental results are loaded without checks.

### Example

See the [demo](https://github.com/Yiannis128/trialist-python/blob/master/demo.ipynb) file.

## Installation

```console
pip install trialist
```

## License

See the [LICENSE](https://github.com/Yiannis128/trialist-python/blob/master/LICENSE) file.
