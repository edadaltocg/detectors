# Detectors

<p align="center">
    <br>
    <img src="_static/face-with-monocle.svg" width="150" height="150" />
    <br>
</p>

Package to accelerate research on out-of-distribution (OOD) detection inspired by Huggingface's transformers.

Under development.

## Stats

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/edadaltocg/detectors/graphs/commit-activity)
[![PyPi version](https://badgen.net/pypi/v/pip/)](https://pypi.org/project/pip)
[![build](https://github.com/edadaltocg/detectors/actions/workflows/python-package.yml/badge.svg)](https://github.com/edadaltocg/detectors/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![PyPI download month](https://img.shields.io/pypi/dm/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![GitHub contributors](https://img.shields.io/github/contributors/Naereen/badges.svg)](https://GitHub.com/Naereen/badges/graphs/contributors/)
[![DOI:](https://zenodo.org/badge/DOI/.svg)](https://doi.org/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.org/project/pip/)

-----

## Features

- Pipelines for evaluating OOD detectors on MNIST, CIFAR, and ImageNet benchmarks.
- Automatic OOD datasets download with md5 checksums.
- Support models implemented at `timm`.
- Models on CIFAR and integrated in `timm`.
- Random seed support for reproducible simulations.
- Several aggregation methods for multi-layer OOD detection.
- OOD detection metrics.
- More than 15 detection methods implemented.
- Pipelines for open set recognition and covariate drift detection.

## Installation

```bash
pip install detectors
```

To install the latest version from the source:

```bash
git clone https://github.com/edadaltocg/detectors.git
cd detectors
pip install -e .
```

## Examples

### Running a benchmark

```python
```

### Optional: Acceleration

This package is compatible with the `accelerate` package to allow for parallel computing.

In order to use it, you need to install it:

```bash
pip install accelerate
```

Then, you can configure it:

```bash
accelerate config
```

And finally, you can run the benchmark with the `accelerate` command:

```bash
accelerate launch demo/ood_benchmark.py
```

#### Configuration example

```text
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 2
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you want to use DeepSpeed? [yes/NO]: NO
Do you want to use FullyShardedDataParallel? [yes/NO]: NO
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all
Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: NO
```

## Changelog

See the [changelog](https://github.com/edadaltocg/detectors/blob/master/CHANGELOG.md) for a history of notable changes to `detectors`.

## Contributing

See the [contributing guidelines](https://github.com/edadaltocg/detectors/blob/master/CONTRIBUTING.md) for instructions on how to contribute to `detectors`.

### Thanks to all our contributors

<a href="https://github.com/edadaltocg/detectors/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=edadaltocg/detectors" />
</a>

### Contact

Concerning this package, its use, and bugs, use the [issue page](https://github.com/edadaltocg/detectors/issues) of the [ruptures repository](https://github.com/edadaltocg/detectors). For other inquiries, you can contact me [here](https://edadaltocg.github.io/contact/).

### Important links

- [Documentation](http://detectors.readthedocs.io/)
- [Pypi package index](https://pypi.python.org/pypi/detectors)

## Citing detectors

```bibtex
@software{detectors,
author = {Dadalto, Eduardo and Colombo, Pierre and Darrin, Maxime and Staerman, Guillaume and Nathan, Noiry and Alberge, Florence and Duhamel, Pierre and Piantanida, Pablo},
month = {3},
title = {{detectors: .}},
url = {https://github.com/edadaltocg/detectors},
version = {0.1.0},
year = {2023}
}
```
