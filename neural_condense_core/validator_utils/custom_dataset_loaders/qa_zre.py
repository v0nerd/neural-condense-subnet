from datasets import load_dataset


def load_qa_zre_dataset():
    r"""
    Load the qa_zre dataset.
    """
    qa_zre = load_dataset("community-datasets/qa_zre", split="train", streaming=True)
    qa_zre = qa_zre.shuffle(seed=42)
    qa_zre = qa_zre.filter(lambda x: len(x["answers"]) > 0)
    qa_zre = qa_zre.map(
        lambda x: {
            "context": x["context"],
            "question": x["question"].replace("XXX", x["subject"]),
            "answer": x["answers"][0],
        }
    )

    return qa_zre
