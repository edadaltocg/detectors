import detectors


def main():
    for dataset_name in detectors.data.list_datasets():
        dataset = detectors.data.create_dataset(dataset_name, download=True)


if __name__ == "__main__":
    main()
