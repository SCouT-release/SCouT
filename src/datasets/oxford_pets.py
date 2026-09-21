from src.datasets.huggingface import HuggingFaceImageClassificationDataset


class OxfordIIITPet(HuggingFaceImageClassificationDataset):
    dataset_id = "timm/oxford-iiit-pet"
    dataset_name = "OxfordIIITPet"


class OxfordPets(OxfordIIITPet):
    dataset_name = "OxfordPets"
