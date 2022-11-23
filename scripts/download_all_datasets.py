import detectors


def main():
    for dataset_name in detectors.data.datasets_registry:
        print(dataset_name)
        dataset = detectors.data.get_dataset(dataset_name, download=True)
        print(dataset)
        print()


if __name__ == "__main__":
    main()
