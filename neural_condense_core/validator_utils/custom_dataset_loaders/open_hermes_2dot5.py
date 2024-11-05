from datasets import load_dataset


def load_open_hermes():
    r"""
    Load the open-hermes-2.5 dataset.
    """
    ds = load_dataset(
        "NurtureAI/OpenHermes-2.5-flattened", split="train", streaming=True
    )
    ds = ds.shuffle(seed=42)
    ds = ds.filter(lambda x: not x["system"])
    ds = ds.map(
        lambda x: {
            "user": x["prompt"],
            "assistant": x["output"],
        }
    )

    return ds
