import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import detectors
from detectors import create_dataset, get_dataset_cls, get_datasets_names
from detectors.config import DATA_DIR, IMAGENET_ROOT
from detectors.data.cifar_wrapper import CIFAR10Wrapped, CIFAR100Wrapped


def test_cifar10():
    transform = transforms.ToTensor()

    cifar10_class = get_dataset_cls("cifar10")
    for split in ("train", "test"):
        dataset = create_dataset("cifar10", root=DATA_DIR, split=split, transform=transform, download=True)
        cifar10_obj = CIFAR10Wrapped(root=DATA_DIR, split=split, transform=transform, download=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert img.shape == (1, 3, 32, 32)
        assert label.shape == (1,)
        assert len(dataset) == len(cifar10_obj)
        assert len(dataset) == 50000 if split == "train" else 10000
    assert issubclass(cifar10_class, torchvision.datasets.CIFAR10)


def test_cifar100():
    transform = transforms.ToTensor()

    cifar100_class = get_dataset_cls("cifar100")
    for split in ("train", "test"):
        dataset = create_dataset("cifar100", root=DATA_DIR, split=split, transform=transform, download=True)
        cifar100_obj = CIFAR100Wrapped(root=DATA_DIR, split=split, transform=transform, download=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert img.shape == (1, 3, 32, 32)
        assert label.shape == (1,)
        assert len(dataset) == len(cifar100_obj)
        assert len(dataset) == 50000 if split == "train" else 10000
    assert issubclass(cifar100_class, torchvision.datasets.CIFAR100)


def test_stl10():
    transform = transforms.ToTensor()

    stl10_class = get_dataset_cls("stl10")
    for split in ("train", "test"):
        dataset = create_dataset("stl10", root=DATA_DIR, split=split, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert label.shape == (1,)
        assert len(dataset) == 5000 if split == "train" else 8000
    assert stl10_class is torchvision.datasets.STL10


def test_svhn():
    transform = transforms.ToTensor()

    svhn_class = get_dataset_cls("svhn")
    for split in ("train", "test"):
        dataset = create_dataset("svhn", root=DATA_DIR, split=split, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert img.shape == (1, 3, 32, 32)
        assert label.shape == (1,)
        assert len(dataset) == 73257 if split == "train" else 26032
    assert svhn_class is torchvision.datasets.SVHN


def test_mnist():
    transform = transforms.ToTensor()

    mnist_class = get_dataset_cls("mnist")
    for split in ("train", "test"):
        dataset = create_dataset("mnist", root=DATA_DIR, split=split, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert img.shape == (1, 1, 28, 28)
        assert label.shape == (1,)
        assert len(dataset) == 60000 if split == "train" else 10000
    assert issubclass(mnist_class, torchvision.datasets.MNIST)


def test_fashion_mnist():
    transform = transforms.ToTensor()

    fmnist_class = get_dataset_cls("fashion_mnist")
    for split in ("train", "test"):
        dataset = create_dataset("fashion_mnist", root=DATA_DIR, split=split, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert img.shape == (1, 1, 28, 28)
        assert label.shape == (1,)
        assert len(dataset) == 60000 if split == "train" else 10000
    assert issubclass(fmnist_class, torchvision.datasets.FashionMNIST)


def test_english_chars():
    transform = transforms.ToTensor()

    english_class = get_dataset_cls("english_chars")
    dataset = create_dataset("english_chars", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert label.shape == (1,)
    assert len(dataset) == 7705
    assert issubclass(english_class, torchvision.datasets.ImageFolder)


def test_isun():
    transform = transforms.ToTensor()

    isun_class = get_dataset_cls("isun")
    dataset = create_dataset("isun", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 8925
    assert issubclass(isun_class, torchvision.datasets.ImageFolder)


def test_lsun_c():
    transform = transforms.ToTensor()

    lsun_class = get_dataset_cls("lsun_c")
    dataset = create_dataset("lsun_c", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000
    assert issubclass(lsun_class, torchvision.datasets.ImageFolder)


def test_lsun_r():
    transform = transforms.ToTensor()

    lsun_class = get_dataset_cls("lsun_r")
    dataset = create_dataset("lsun_r", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000
    assert issubclass(lsun_class, torchvision.datasets.ImageFolder)


def test_tiny_imagenet_c():
    transform = transforms.ToTensor()

    tiny_imagenet_c_class = get_dataset_cls("tiny_imagenet_c")
    dataset = create_dataset("tiny_imagenet_c", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000
    assert issubclass(tiny_imagenet_c_class, torchvision.datasets.ImageFolder)


def test_tiny_imagenet_r():
    transform = transforms.ToTensor()

    tiny_imagenet_r_class = get_dataset_cls("tiny_imagenet_r")
    dataset = create_dataset("tiny_imagenet_r", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000
    assert issubclass(tiny_imagenet_r_class, torchvision.datasets.ImageFolder)


def test_textures():
    transform = transforms.ToTensor()

    textures_class = get_dataset_cls("textures")
    dataset = create_dataset("textures", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 5640
    assert issubclass(textures_class, torchvision.datasets.ImageFolder)


def test_gaussian():
    transform = transforms.ToTensor()

    dataset = create_dataset("gaussian", root=DATA_DIR, split=None, transform=transform, nb_samples=10000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000


def test_uniform():
    transform = transforms.ToTensor()

    dataset = create_dataset("uniform", root=DATA_DIR, split=None, transform=transform, nb_samples=10000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000


# def test_places365():
#     transform = transforms.ToTensor()

#     places365_class = get_dataset_cls("places365")
#     dataset = create_dataset("places365", root=DATA_DIR, split="val", transform=transform, download=True)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#     img, label = next(iter(dataloader))

#     assert type(img) == torch.Tensor
#     assert len(dataset) == 10000
#     assert places365_class is torchvision.datasets.Places365


def test_stanford_cars():
    transform = transforms.ToTensor()

    stanford_cars_class = get_dataset_cls("stanford_cars")
    for split in ("train", "test"):
        dataset = create_dataset("stanford_cars", root=DATA_DIR, split="train", transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert type(img) == torch.Tensor
        assert len(dataset) == 8144 if split == "train" else 8041
    assert stanford_cars_class is torchvision.datasets.StanfordCars


def test_mos_inaturalist():
    transform = transforms.ToTensor()

    mos_inaturalist_class = get_dataset_cls("mos_inaturalist")

    dataset = create_dataset("mos_inaturalist", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000
    assert issubclass(mos_inaturalist_class, torchvision.datasets.ImageFolder)


def test_mos_places365():
    transform = transforms.ToTensor()

    mos_places365_class = get_dataset_cls("mos_places365")

    dataset = create_dataset("mos_places365", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000
    assert issubclass(mos_places365_class, torchvision.datasets.ImageFolder)


def test_mos_sun():
    transform = transforms.ToTensor()

    mos_sun_class = get_dataset_cls("mos_sun")

    dataset = create_dataset("mos_sun", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 10000
    assert issubclass(mos_sun_class, torchvision.datasets.ImageFolder)


def test_imagenet_o():
    transform = transforms.ToTensor()

    imagenet_o_class = get_dataset_cls("imagenet_o")

    dataset = create_dataset("imagenet_o", root=DATA_DIR, split=None, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 2000
    assert issubclass(imagenet_o_class, torchvision.datasets.ImageFolder)


def test_imagenet():
    transform = transforms.ToTensor()

    imagenet_class = get_dataset_cls("imagenet")

    dataset = create_dataset("imagenet", root=IMAGENET_ROOT, split="val", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img, label = next(iter(dataloader))

    assert type(img) == torch.Tensor
    assert len(dataset) == 50000
    assert imagenet_class is torchvision.datasets.ImageNet


def test_cifar10c():
    transform = transforms.ToTensor()

    cifar10c_class = get_dataset_cls("cifar10c")

    for intensity in [1, 2, 3, 4, 5]:
        dataset = create_dataset(
            "cifar10c", root=DATA_DIR, split="gaussian_blur", intensity=intensity, transform=transform
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert type(img) == torch.Tensor
        assert len(dataset) == 10000
        assert issubclass(cifar10c_class, torch.utils.data.Dataset)


def test_cifar100c():
    transform = transforms.ToTensor()

    cifar10c_class = get_dataset_cls("cifar100c")

    for intensity in [1, 2, 3, 4, 5]:
        dataset = create_dataset(
            "cifar100c", root=DATA_DIR, split="gaussian_blur", intensity=intensity, transform=transform
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        img, label = next(iter(dataloader))

        assert type(img) == torch.Tensor
        assert len(dataset) == 10000
        assert issubclass(cifar10c_class, torch.utils.data.Dataset)


if __name__ == "__main__":
    test_cifar10c()
    test_cifar100c()
