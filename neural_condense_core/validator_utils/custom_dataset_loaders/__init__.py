from .qa_zre import load_qa_zre_dataset
from .open_hermes_2dot5 import load_open_hermes


def load_custom_dataset(dataset_name: str):
    r"""
    Load the custom dataset.
    Args:
    - dataset_name (str): The name of the dataset.
    """
    if dataset_name == "qa_zre":
        return load_qa_zre_dataset()
    elif dataset_name == "open_hermes":
        return load_open_hermes()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
