"""Example on how to list all resources available in `detectors` package."""

import detectors

if __name__ == "__main__":
    # list all available models
    print("Models:")
    print(detectors.list_models())
    print(detectors.list_models("*cifar*"))

    # list all available datasets
    print("Datasets:")
    print(detectors.list_datasets())

    # list all available detectors
    print("Detectors:")
    print(detectors.list_detectors())

    # list all available pipelines
    print("Pipelines:")
    print(detectors.list_pipelines())
