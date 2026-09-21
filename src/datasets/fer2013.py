from src.datasets.huggingface import HuggingFaceImageClassificationDataset


class FER2013(HuggingFaceImageClassificationDataset):
    dataset_id = "AutumnQiu/fer2013"
    dataset_name = "FER2013"


class FER2013Val(FER2013):
    test_split = "valid"
