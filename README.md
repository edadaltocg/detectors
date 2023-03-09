<p align="center">
    <br>
    <img src="_static/face-with-monocle.svg" width="150" height="150" />
    <br>
</p>

# Detectors

Package to accelerate research on out-of-distribution (OOD) detection inspired by Huggingface's transformers.

Under development.

## Deployment & Documentation & Stats & License

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/edadaltocg/detectors/graphs/commit-activity)
[![PyPi version](https://badgen.net/pypi/v/pip/)](https://pypi.org/project/pip)
[![build](https://github.com/edadaltocg/detectors/actions/workflows/python-package.yml/badge.svg)](https://github.com/edadaltocg/detectors/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![GitHub stars](https://img.shields.io/github/stars/edadaltocg/detectors.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/edadaltocg/detectors/stargazers/)
[![GitHub forks](https://badgen.net/github/forks/edadaltocg/detectors/)](https://github.com/edadaltocg/detectors/network/)
[![PyPI download month](https://img.shields.io/pypi/dm/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![GitHub contributors](https://img.shields.io/github/contributors/Naereen/badges.svg)](https://GitHub.com/Naereen/badges/graphs/contributors/)
[![DOI:](https://zenodo.org/badge/DOI/.svg)](https://doi.org/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.org/project/pip/)

-----

## Table of Contents

<!--
## Examples

```python
``` -->

## Features

- Datasets with md5 checksums.
- Models pre-trained on CIFAR and integrated in `timm`.
- Pipeline for evaluating OOD detectors on CIFAR and ImageNet benchmarks.
- Multiple seed simulations.
- Diverse aggregation methods for OOD detection.
- Efficient OOD detection metrics.

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

## Running a benchmark

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

## Thanks to all our contributors

<a href="https://github.com/edadaltocg/detectors/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=edadaltocg/detectors" />
</a>

### Contact

Concerning this package, its use and bugs, use the [issue page](https://github.com/edadaltocg/detectors/issues) of the [ruptures repository](https://github.com/edadaltocg/detectors). For other inquiries, you can contact me [here](https://edadaltocg.github.io/contact/).

### Important links

- [Documentation](http://detectors.readthedocs.io/)
- [Pypi package index](https://pypi.python.org/pypi/detectors)

### Changelog

See the [changelog](https://github.com/edadaltocg/detectors/blob/master/CHANGELOG.md) for a history of notable changes to `detectors`.

## Contributing

See the [contributing guidelines](https://github.com/edadaltocg/detectors/blob/master/CONTRIBUTING.md) for instructions on how to contribute to `detectors`.

<!-- ## TODO

- Pipeline for generating results table.

## Citing detectors

```bibtex
``` -->