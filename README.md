# 🧐 Detectors

Package to accelerate research on generalized out-of-distribution (OOD) detection.

Under development. Please report any issues or bugs [here](https://github.com/edadaltocg/detectors/issues).

## Stats

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/edadaltocg/detectors/graphs/commit-activity)
[![build](https://github.com/edadaltocg/detectors/actions/workflows/build.yml/badge.svg)](https://github.com/edadaltocg/detectors/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/detectors/badge/?version=latest)](http://detectors.readthedocs.io/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7883080.svg)](https://doi.org/10.5281/zenodo.7883080)

## What is it?

This library is aimed at assisting researchers in the field of generalized OOD detection. It is inspired by [HF's Transformers](https://https://github.com/huggingface/transformers) and features implementations of baselines, metrics, and data sets that allow researchers to perform meaningful benchmarking and development of ood detection methods. It features:

- `methods`: more than 20 detection methods implemented.
- `pipelines`: evaluating OOD detectors on popular benchmarks, such as MNIST, CIFAR, and ImageNet benchmarks with random seed support for reproducibility.
- `datasets`: OOD datasets implemented with md5 checksums and without the need to download them manually.
- `models`: model architectures totally integrated with [`timm`](https://github.com/huggingface/pytorch-image-models).
- `eval`: implementation of fast OOD evaluation metrics.
- Several aggregation methods for multi-layer OOD detection.
- Pipelines for open set recognition and covariate drift detection.

## Installation

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch. Installing PyTorch with CUDA support is strongly recommended.

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

## FAQ over specific documents

**Methods**

- [Documentation](https://detectors.readthedocs.io/en/latest/use_cases/)

**Pipelines**

- [Documentation](https://detectors.readthedocs.io/en/latest/use_cases/)

**Pypi**

- [Website](https://pypi.org/project/detectors)

## Contributing

As an open-source project in a rapidly developing field, we are open to contributions, whether in the form of a new feature, improved infra, or better documentation.

See the [contributing guidelines](https://github.com/edadaltocg/detectors/blob/master/CONTRIBUTING.md) for instructions on how to make your first contribution to `detectors`.

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

The detection of Out-of-Distribution (OOD) has created a new way of securing machine intelligence, but despite its many successes, it can be difficult to understand due to the various methods available and their intricate implementations. The fast pace of research and the wide range of OOD methods makes it challenging to navigate the field, which can be a problem for those who have recently joined the field or want to deploy OOD detection. The library we have created aims to lower these barriers by providing a resource for researchers of any background to understand the methods available, how they work, and how to be successful with OOD detection.

If you find this repository useful, please consider giving it a star 🌟 and citing it as below:

```bibtex
@software{detectors2023,
author = {Eduardo Dadalto},
title = {Detectors: a Python Library for Generalized Out-Of-Distribution Detection},
url = {https://github.com/edadaltocg/detectors},
doi = {https://doi.org/10.5281/zenodo.7883596},
month = {5},
year = {2023}
}
```