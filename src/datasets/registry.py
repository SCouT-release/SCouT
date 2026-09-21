import copy

import torch
from torch.utils.data.dataset import random_split

from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.fer2013 import FER2013, FER2013Val
from src.datasets.flowers102 import Flowers102, Flowers102Val
from src.datasets.gtsrb import GTSRB
from src.datasets.imagenet import ImageNet
from src.datasets.mnist import MNIST
from src.datasets.oxford_pets import OxfordIIITPet, OxfordPets
from src.datasets.pcam import PCAM, PCAMVal
from src.datasets.resisc45 import RESISC45
from src.datasets.stl10 import STL10
from src.datasets.sun397 import SUN397
from src.datasets.svhn import SVHN

registry = {
    dataset_class.__name__: dataset_class
    for dataset_class in (
        Cars,
        CIFAR10,
        CIFAR100,
        DTD,
        EuroSAT,
        EuroSATVal,
        FER2013,
        FER2013Val,
        Flowers102,
        Flowers102Val,
        GTSRB,
        ImageNet,
        MNIST,
        OxfordIIITPet,
        OxfordPets,
        PCAM,
        PCAMVal,
        RESISC45,
        STL10,
        SUN397,
        SVHN,
    )
}


class GenericDataset:
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(
    dataset,
    new_dataset_class_name,
    batch_size,
    num_workers,
    val_fraction,
    max_val_samples=None,
    seed=0,
):
    assert 0 < val_fraction < 1
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    trainset, valset = random_split(
        dataset.train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    if new_dataset_class_name == "MNISTVal":
        assert trainset.indices[0] == 36044

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()
    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    new_dataset.classnames = copy.copy(dataset.classnames)
    return new_dataset


# For names ending in `Val`, use an explicit validation class when available;
# otherwise split the original training set deterministically.
def get_dataset(
    dataset_name,
    preprocess,
    location,
    batch_size=128,
    num_workers=16,
    val_fraction=0.1,
    max_val_samples=5000,
):
    if dataset_name.endswith("Val"):
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.removesuffix("Val")
            base_dataset = get_dataset(
                base_dataset_name,
                preprocess,
                location,
                batch_size,
                num_workers,
            )
            return split_train_into_train_val(
                base_dataset,
                dataset_name,
                batch_size,
                num_workers,
                val_fraction,
                max_val_samples,
            )
    else:
        supported = ", ".join(sorted(registry))
        assert dataset_name in registry, (
            f"Unsupported dataset: {dataset_name}. Supported datasets: {supported}"
        )
        dataset_class = registry[dataset_name]

    return dataset_class(
        preprocess,
        location=location,
        batch_size=batch_size,
        num_workers=num_workers,
    )
