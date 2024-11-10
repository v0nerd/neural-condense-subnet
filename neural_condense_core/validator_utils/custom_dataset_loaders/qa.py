from datasets import load_dataset
import random


def load_squad_dataset():
    r"""
    Load the qa_zre dataset.
    """

    def _format_squad(x):
        context = x["context"]
        question = (
            x["question"]
            + "If the question is unanswerable, please type 'Unanswerable'."
        )
        answer = x["answers"]["text"] or ["Unanswerable"]
        answer = random.choice(answer)
        return {"context": context, "question": question, "answer": answer}

    ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=True)
    ds = ds.map(_format_squad)

    return ds


def load_coqa_dataset():
    r"""
    Load the qa_zre dataset.
    """

    def _format_coqa(x):
        context = x["story"]
        questions = x["questions"]
        answers = x["answers"]["input_text"]
        q_a_pairs = zip(questions, answers)
        question, answer = random.choice(list(q_a_pairs))
        return {"context": context, "question": question, "answer": answer}

    ds = load_dataset("stanfordnlp/coqa", split="train", streaming=True)
    ds = ds.map(_format_coqa)
    return ds
