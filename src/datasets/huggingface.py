import os

import torch


class HuggingFaceImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item["image"].convert("RGB")
        label = item["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def huggingface_cache_dir(location):
    return os.path.join(os.path.expanduser(location), "huggingface")


def load_huggingface_split(dataset_id, split, location, dataset_name):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            f"{dataset_name} loading from Hugging Face requires the `datasets` "
            "package. Install it with `pip install datasets`."
        ) from exc

    return load_dataset(
        dataset_id,
        split=split,
        cache_dir=huggingface_cache_dir(location),
    )


def get_huggingface_classnames(hf_dataset):
    label_feature = hf_dataset.features["label"]
    if hasattr(label_feature, "names"):
        names = label_feature.names
        if isinstance(names, dict):
            names = [names[str(idx)] for idx in range(len(names))]
        return [str(name).replace("_", " ").lower() for name in names]

    labels = sorted(set(hf_dataset["label"]))
    return [str(label) for label in labels]


class HuggingFaceImageClassificationDataset:
    dataset_id = None
    dataset_name = None
    train_split = "train"
    test_split = "test"
    classnames = None

    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=16,
    ):
        print(f"Loading {self.dataset_id} from Hugging Face.")
        train_hf = load_huggingface_split(
            self.dataset_id, self.train_split, location, self.dataset_name
        )
        test_hf = load_huggingface_split(
            self.dataset_id, self.test_split, location, self.dataset_name
        )

        self.train_dataset = HuggingFaceImageDataset(train_hf, preprocess)
        self.test_dataset = HuggingFaceImageDataset(test_hf, preprocess)
        self.classnames = self.classnames or get_huggingface_classnames(train_hf)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
