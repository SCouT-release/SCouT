import os
import re

import torch


def pretify_classname(classname):
    if " " in classname or "_" in classname:
        out = classname.replace("_", " ").lower()
    else:
        words = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", classname)
        out = " ".join(word.lower() for word in words)

    if out.endswith("al"):
        return out + " area"
    return out


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
            "EuroSAT loading from Hugging Face requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from exc

    cache_dir = os.path.join(os.path.expanduser(location), "huggingface")
    return load_dataset("tanganke/eurosat", split=split, cache_dir=cache_dir)


def _get_huggingface_classnames(hf_dataset):
    label_feature = hf_dataset.features["label"]
    if hasattr(label_feature, "names"):
        classnames = label_feature.names
    else:
        classnames = [str(label) for label in sorted(set(hf_dataset["label"]))]

    ours_to_open_ai = {
        "annual crop": "annual crop land",
        "forest": "forest",
        "herbaceous vegetation": "brushland or shrubland",
        "highway": "highway or road",
        "industrial area": "industrial buildings or commercial buildings",
        "pasture": "pasture land",
        "permanent crop": "permanent crop land",
        "residential area": "residential buildings or homes or apartments",
        "river": "river",
        "sea lake": "lake or sea",
    }

    classnames = [pretify_classname(classname) for classname in classnames]
    return [ours_to_open_ai.get(classname, classname) for classname in classnames]


class EuroSATBase:
    def __init__(
        self,
        preprocess,
        test_split,
        location="~/datasets",
        batch_size=32,
        num_workers=16,
    ):
        print("Loading tanganke/eurosat from Hugging Face.")
        location = os.path.expanduser(location)
        train_hf = _load_huggingface_split("train", location)

        if test_split == "val":
            split_hf = train_hf.train_test_split(test_size=0.25, seed=0)
            train_hf = split_hf["train"]
            test_hf = split_hf["test"]
        else:
            test_hf = _load_huggingface_split(test_split, location)

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


class EuroSAT(EuroSATBase):
    def __init__(
        self,
        preprocess,
        location="~/datasets",
        batch_size=32,
        num_workers=16,
    ):
        super().__init__(preprocess, "test", location, batch_size, num_workers)


class EuroSATVal(EuroSATBase):
    def __init__(
        self,
        preprocess,
        location="~/datasets",
        batch_size=32,
        num_workers=16,
    ):
        super().__init__(preprocess, "val", location, batch_size, num_workers)
