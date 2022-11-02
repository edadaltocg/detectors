from detectors.data import get_dataset, get_dataset_cls, get_datasets_names


def test_download_data():
    for dataset_name in get_datasets_names():
        print(dataset_name)
        get_dataset_cls(dataset_name)
        if dataset_name in ["imagenet1k", "ilsvrc2012"]:
            get_dataset(dataset_name, split="val")
        else:
            get_dataset(dataset_name, split="test", download=True)


def test_data_loading():
    for dataset_name in get_datasets_names():
        print(dataset_name)
        get_dataset_cls(dataset_name)
        if dataset_name in ["imagenet1k", "ilsvrc2012"]:
            dataset = get_dataset(dataset_name, split="val")
        else:
            dataset = get_dataset(dataset_name, split="test", download=True)

        for img, label in dataset:
            break


if __name__ == "__main__":
    test_download_data()
    test_data_loading()
