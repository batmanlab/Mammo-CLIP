from .image_aligner import ImageAligenerDataset
from .image_classification import ImageClassificationDataset
from .image_classification_zs import ImageClassificationZSDataset
from .imagetext import ImageTextDataset
from .imagetext_contrastive import ImageTextDataset_contrastive
from .imagetext_retrieval import ImageTextDataset_Retrieval


def load_dataset(data_type: str, loss_config=None, transform_config=None, **kwargs):
    if data_type.lower() == "imagetext":
        dataset = ImageTextDataset(loss_config=loss_config, transform_config=transform_config, **kwargs)
    elif data_type.lower() == "imagetext_contrastive":
        dataset = ImageTextDataset_contrastive(loss_config=loss_config, transform_config=transform_config, **kwargs)
    elif data_type.lower() == "imagetext_retrieval":
        dataset = ImageTextDataset_Retrieval(loss_config=loss_config, transform_config=transform_config, **kwargs)
    elif data_type.lower() == "image_classification":
        dataset = ImageClassificationDataset(transform_config=transform_config, **kwargs)
    elif data_type.lower() == "image_classification_zs":
        dataset = ImageClassificationZSDataset(transform_config=transform_config, **kwargs)
    elif data_type.lower() == "image_aligner":
        dataset = ImageAligenerDataset(transform_config=transform_config, **kwargs)
    else:
        raise KeyError(f"Not supported data type: {data_type}")
    return dataset
