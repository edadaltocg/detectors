from typing import Callable, Optional

import wilds

datasets_info = {
    "iwildcam": {
        "modality": "Image",
        "splits": ["train", "val", "test", "id_val", "id_test"],
        "unlabeled_splits": ["extra_unlabeled"],
    },
    "camelyon17": {
        "modality": "Image",
        "splits": ["train", "val", "test", "id_val"],
        "unlabeled_splits": ["train_unlabeled", "val_unlabeled", "test_unlabeled"],
    },
    "rxrx1": {
        "modality": "Image",
        "splits": ["train", "val", "test", "id_test"],
        "unlabeled_splits": [],
    },
    "ogb-molpcba": {
        "modality": "Graph",
        "splits": ["train", "val", "test"],
        "unlabeled_splits": ["train_unlabeled", "val_unlabeled", "test_unlabeled"],
    },
    "globalwheat": {
        "modality": "Image",
        "splits": ["train", "val", "test", "id_val", "id_test"],
        "unlabeled_splits": ["train_unlabeled", "val_unlabeled", "test_unlabeled", "extra_unlabeled"],
    },
    "civilcomments": {
        "modality": "Text",
        "splits": ["train", "val", "test"],
        "unlabeled_splits": ["extra_unlabeled"],
    },
    "fmow": {
        "modality": "Image",
        "splits": ["train", "val", "test", "id_val", "id_test"],
        "unlabeled_splits": ["train_unlabeled", "val_unlabeled", "test_unlabeled"],
    },
    "poverty": {
        "modality": "Image",
        "splits": ["train", "val", "test", "id_val", "id_test"],
        "unlabeled_splits": ["train_unlabeled", "val_unlabeled", "test_unlabeled"],
    },
    "amazon": {
        "modality": "Text",
        "splits": ["train", "val", "test", "id_val", "id_test"],
        "unlabeled_splits": ["val_unlabeled", "test_unlabeled", "extra_unlabeled"],
    },
    "py150": {
        "modality": "Text",
        "splits": ["train", "val", "test", "id_val", "id_test"],
        "unlabeled_splits": [],
    },
}


def make_wilds_dataset(
    dataset_name, root, split="train", transform: Optional[Callable] = None, download=False, **kwargs
):
    dataset = wilds.get_dataset(dataset_name, root_dir=root, download=download)
    assert dataset is not None
    dataset = dataset.get_subset(split, transform=transform)

    return dataset


if __name__ == "__main__":
    import detectors.config as cfg

    for ds_name in datasets_info.keys():
        print(f"Creating {ds_name} dataset")
        ds = make_wilds_dataset(ds_name, root=cfg.DATA_DIR, split="train", download=True)
