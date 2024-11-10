from datasets import load_dataset


def _format_messages(x):
    conversations: list[dict[str, str]] = x["conversations"]
    conversations = conversations[:2]
    messages = {}
    for conversation in conversations:
        if conversation["from"] == "human":
            messages["user"] = conversation["value"]
        else:
            messages["assistant"] = conversation["value"]
    return messages


def load_infinity_instruct():
    r"""
    Load the Infinity Instruct dataset.
    """
    ds = load_dataset(
        "manifoldlabs/Infinity-Instruct", "7M", split="train", streaming=True
    )
    ds = ds.filter(lambda x: x["langdetect"] == "en")
    ds = ds.map(_format_messages)
    return ds
