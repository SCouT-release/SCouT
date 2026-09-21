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


def _load_huggingface_split(split, location):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "SUN397 loading from Hugging Face requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from exc

    cache_dir = os.path.join(os.path.expanduser(location), "huggingface")
    return load_dataset("tanganke/sun397", split=split, cache_dir=cache_dir)


def _get_huggingface_classnames(hf_dataset):
    label_feature = hf_dataset.features["label"]
    if hasattr(label_feature, "names"):
        return [name.replace("_", " ") for name in label_feature.names]

    labels = sorted(set(hf_dataset["label"]))
    return [str(label) for label in labels]


class SUN397:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=16,
    ):
        print("Loading tanganke/sun397 from Hugging Face.")
        train_hf = _load_huggingface_split("train", location)
        test_hf = _load_huggingface_split("test", location)

        self.train_dataset = HuggingFaceImageDataset(train_hf, preprocess)
        self.test_dataset = HuggingFaceImageDataset(test_hf, preprocess)
        self.classnames = _get_huggingface_classnames(train_hf)

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
