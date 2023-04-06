# Examples

## Creating a detector

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

## Running a benchmark

The following example shows how to run a benchmark.

```python
import detectors


model = detectors.create_model("resnet18_cifar10", pretrained=True)
test_transform = detectors.create_transform(model)

pipeline = detectors.create_pipeline("ood_benchmark_cifar10", transform=test_transform)
method = detectors.create_detector("awesome_detector", model=model)

pipeline_results = pipeline.run(method)
print(pipeline.report(pipeline_results["results"]))
```

## Listing available resources

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
