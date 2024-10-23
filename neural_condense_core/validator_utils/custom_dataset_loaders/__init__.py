from .qa_zre import load_qa_zre_dataset


def load_custom_dataset(dataset_name: str):
    r"""
    Load the custom dataset.
    Args:
    - dataset_name (str): The name of the dataset.
    """
    if dataset_name == "qa_zre":
        return load_qa_zre_dataset()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
