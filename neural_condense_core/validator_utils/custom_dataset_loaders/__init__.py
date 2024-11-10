from .qa import load_coqa_dataset, load_squad_dataset
from .open_infinity_instruct import load_infinity_instruct
from .open_math_instruct import load_open_math_instruct_dataset


def load_custom_dataset(dataset_name: str):
    r"""
    Load the custom dataset.
    Args:
    - dataset_name (str): The name of the dataset.
    """
    if dataset_name == "squad_v2":
        return load_squad_dataset()
    elif dataset_name == "coqa":
        return load_coqa_dataset()
    elif dataset_name == "infinity_instruct":
        return load_infinity_instruct()
    elif dataset_name == "open_math_instruct":
        return load_open_math_instruct_dataset()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
