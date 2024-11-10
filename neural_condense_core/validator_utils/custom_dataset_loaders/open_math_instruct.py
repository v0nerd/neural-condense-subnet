from datasets import load_dataset


def load_open_math_instruct_dataset():
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    ds = ds.map(
        lambda x: {
            "user": x["problem"],
            "assistant": x["generated_solution"],
        }
    )

    return ds
