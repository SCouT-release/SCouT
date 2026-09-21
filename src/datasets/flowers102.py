from src.datasets.huggingface import HuggingFaceImageClassificationDataset


class Flowers102(HuggingFaceImageClassificationDataset):
    dataset_id = "dpdl-benchmark/oxford_flowers102"
    dataset_name = "Flowers102"


class Flowers102Val(Flowers102):
    test_split = "validation"
