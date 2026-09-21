from src.datasets.huggingface import HuggingFaceImageClassificationDataset


class PCAM(HuggingFaceImageClassificationDataset):
    dataset_id = "anurag2op/pcam-augmented-train-test-val-50k"
    dataset_name = "PCAM"


class PCAMVal(PCAM):
    test_split = "validation"
