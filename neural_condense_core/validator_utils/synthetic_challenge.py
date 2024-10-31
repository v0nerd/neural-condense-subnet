from datasets import load_dataset
from ..protocol import TextCompressProtocol
from transformers import AutoTokenizer
import re
from .custom_dataset_loaders import load_custom_dataset
from typing import Iterator


class Challenger:
    def __init__(self):
        self.raw_dataset = self._load_raw_dataset()
        self.qa_dataset = self._load_qa_dataset()
        self.sat = "[START-ACTIVATE-TOKEN]"
        self.eat = "[END-ACTIVATE-TOKEN]"

    def __call__(
        self, tokenizer: AutoTokenizer, task: str = ""
    ) -> TextCompressProtocol:
        r"""
        Get a sample for the given task.
        Args:
        - task (str): The task.
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        """
        if task == "qa":
            return self._get_qa_sample(tokenizer)
        else:
            return self._get_ae_sample(tokenizer)

    def _get_qa_sample(self, tokenizer: AutoTokenizer) -> TextCompressProtocol:
        r"""
        Get a sample for the qa task.
        In this task, the assistant is asked to answer a question given the context.
        Args:
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        """
        item = next(self.qa_dataset)
        contexts = item["contexts"]
        question = item["question"]
        answer = item["answer"]

        prompt_after_context = f"\n\nAnswer the following question: {question}"

        contexts_as_bullets = "\n".join([f"- {context}" for context in contexts])

        messages = [
            {
                "role": "user",
                "content": f"""Given the context: {contexts_as_bullets}.{self.sat} {prompt_after_context}{self.eat}""",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        messages_str = tokenizer.apply_chat_template(messages, tokenize=False)

        context, activation_prompt, expected_completion = re.split(
            f"{re.escape(self.sat)}|{re.escape(self.eat)}", messages_str
        )

        protocol = TextCompressProtocol(
            context=context,
            expected_completion=expected_completion,
            activation_prompt=activation_prompt,
        )
        return protocol

    def _get_ae_sample(self, tokenizer: AutoTokenizer) -> TextCompressProtocol:
        r"""
        Get a sample for the autoencoder task.
        In this task, the assistant is asked to rewrite the same text given the context.
        Args:
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        - k (int): The length of the text to be compressed.
        """
        text = next(self.raw_dataset)["text"]
        prompt_after_context = "\n\nRewrite exactly the same text."
        messages = [
            {
                "role": "user",
                "content": f"""Given the context: {text}.{self.sat} {prompt_after_context}{self.eat}""",
            },
            {
                "role": "assistant",
                "content": text,
            },
        ]

        messages_str = tokenizer.apply_chat_template(messages, tokenize=False)

        context, activation_prompt, expected_completion = re.split(
            f"{re.escape(self.sat)}|{re.escape(self.eat)}", messages_str
        )

        protocol = TextCompressProtocol(
            context=context,
            expected_completion=expected_completion,
            activation_prompt=activation_prompt,
        )
        return protocol

    def _load_qa_dataset(self) -> Iterator[dict]:
        r"""
        Combine multiples qa dataset.
        Return a combined dataset that has following schema:
        ------------------------------
        | contexts | question | answer |
        ------------------------------
        """
        datasets = [
            load_custom_dataset("qa_zre"),
        ]

        def generator():
            n_contexts = 20
            contexts = []
            for dataset in datasets:
                for example in dataset:
                    if len(contexts) < n_contexts:
                        contexts.append(example["context"])
                        continue
                    yield {
                        "contexts": contexts,
                        "question": example["question"],
                        "answer": example["answer"],
                    }

        return generator()

    def _load_raw_dataset(self) -> Iterator[dict]:
        r"""
        Load the raw dataset for ae task.
        """
        raw_dataset = load_dataset(
            "gair-prox/FineWeb-pro", streaming=True, split="train"
        )
        raw_dataset = raw_dataset.shuffle()
        raw_dataset = iter(raw_dataset)
        return raw_dataset
