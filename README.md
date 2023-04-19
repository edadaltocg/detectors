# 🧐 Detectors

Package to accelerate research on out-of-distribution (OOD) detection.

## Stats

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/edadaltocg/detectors/graphs/commit-activity)
[![build](https://github.com/edadaltocg/detectors/actions/workflows/build.yml/badge.svg)](https://github.com/edadaltocg/detectors/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/detectors/badge/?version=latest)](http://detectors.readthedocs.io/?badge=latest)
[![PyPI download month](https://img.shields.io/pypi/dm/detectors.svg)](https://pypi.python.org/pypi/detectors/)
[![DOI:](https://zenodo.org/badge/DOI/.svg)](https://doi.org/)
![AUR license](https://img.shields.io/aur/license/detectors)

## What is it?

This library is aimed at assisting researchers in the field of generalized OOD detection. It is inspired by [HF's Transformers](https://https://github.com/huggingface/transformers) and provides a set of tools to run benchmarks, evaluate detectors, and compare them. It includes:

- More than 15 detection methods implemented.
- Pipelines for evaluating OOD detectors on MNIST, CIFAR, and ImageNet benchmarks.
- Automatic OOD datasets download with md5 checksums.
- Support models implemented at [`timm`](https://github.com/huggingface/pytorch-image-models).
- Models on CIFAR integrated at `timm`.
- Random seed support for reproducible simulations.
- Implementation of fast OOD evaluation metrics.
- Several aggregation methods for multi-layer OOD detection.
- Pipelines for open set recognition and covariate drift detection.

## Installation

```bash
pip install detectors
```

To install the latest version from the source:

```bash
git clone https://github.com/edadaltocg/detectors.git
cd detectors
pip install --upgrade pip setuptools wheel
pip install -e .
```

Also, you have easy access to the Python scripts from the examples:

```bash
cd examples
```

## Examples

The following examples show how to use the library and how it can be integrated into your research. For more examples, please check the [documentation](https://detectors.readthedocs.io/en/latest/use_cases/).

### Running a benchmark

The following example shows how to run a benchmark.

```python
import detectors
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detectors.create_model("resnet18_cifar10", pretrained=True)
model = model.to(device)
test_transform = detectors.create_transform(model)

pipeline = detectors.create_pipeline("ood_benchmark_cifar10", transform=test_transform)
method = detectors.create_detector("msp", model=model)

pipeline_results = pipeline.run(method)
print(pipeline.report(pipeline_results["results"]))
```

We recommend running benchmarks on machines equipped with large RAM and GPUs with 16GB of memory or larger to leverage large batch sizes and faster inference.

### Creating a detector

The following example shows how to create a detector. The only requirement is that the method takes an input `x` and returns a score.

```python
import torch
import detectors


@detectors.register_detector("awesome_detector")
def awesome_detector(x: torch.Tensor, model, **kwargs):
    # Do something awesome with the model and the input
    return scores

# Instantiating the detector
method = detectors.create_detector("awesome_detector", model=model)
```

Alternatively, you can use the `Detector` class to create a detector that requires some initialization or state to be fitted before being called (e.g., Mahalanobis detector):

```python
import torch
import detectors


@detectors.register_detector("awesome_detector")
class AwesomeDetector(detectors.Detector):
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, x: torch.Tensor, **kwargs):
        # Do something awesome with the model and the input
        return scores

# Instantiating the detector
method = detectors.create_detector("awesome_detector", model=model)
```

Check the [documentation](https://detectors.readthedocs.io/en/latest/use_cases/) for more information.

### Listing available resources

The following example shows how to list all available resources in the library.

```python
import detectors


# list all available models (same as timm.list_models)
print(detectors.list_models())
# list all available models with a specific pattern
print(detectors.list_models("*cifar*"))
# list all available datasets
print(detectors.list_datasets())
# list all available detectors
print(detectors.list_detectors())
# list all available pipelines
print(detectors.list_pipelines())
```

### Optional: Pipeline acceleration

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

## FAQ over specific documents

**Methods**

- [Documentation](https://detectors.readthedocs.io/en/latest/use_cases/)

**Pipelines**

- [Documentation](https://detectors.readthedocs.io/en/latest/use_cases/)

## Contributing

As an open source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infra, or better documentation.

See the [contributing guidelines](https://github.com/edadaltocg/detectors/blob/master/CONTRIBUTING.md) for instructions on how to make your first contribution to `detectors`

### Thanks to all our contributors

<a href="https://github.com/edadaltocg/detectors/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=edadaltocg/detectors" />
</a>

## Contact

Concerning this package, its use, and bugs, use the [issue page](https://github.com/edadaltocg/detectors/issues) of the [ruptures repository](https://github.com/edadaltocg/detectors). For other inquiries, you can contact me [here](https://edadaltocg.github.io/contact/).

## Important links

- [Documentation](http://detectors.readthedocs.io/)
- [Pypi package index](https://pypi.python.org/pypi/detectors)
- [Github repository](https://github.com/edadaltocg/detectors)

## Limitations

- This library is only compatible with PyTorch models.
- This library has implemented only computer vision pipelines and datasets.

## Citing detectors

If you use this library, please cite it as below:

```bibtex
@software{detectors,
author = {Dadalto, Eduardo and Colombo, Pierre and Darrin, Maxime and Staerman, Guillaume and and Alberge, Florence and Duhamel, Pierre and Piantanida, Pablo},
title = {Detectors: generalized out-of-distribution detection library},
url = {https://github.com/edadaltocg/detectors},
month = {3},
year = {2023}
}
```
