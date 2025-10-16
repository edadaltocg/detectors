import time

import torch

import detectors
from detectors import create_detector, create_model, create_pipeline, create_transform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model("resnet18_cifar10", pretrained=False).to(device)
    transform = create_transform(model)

    pipeline = create_pipeline("ood_benchmark_cifar10", transform=transform, num_workers=0)

    method = create_detector("cf_ood", model=model, alpha=0.1, k=0.01)

    t0 = time.time()
    results = pipeline.run(method)
    print("Done in", round(time.time() - t0, 2), "s")
    print(pipeline.report(results["results"]))


if __name__ == "__main__":
    main()
