from datasets import load_dataset, IterableDataset
from ..protocol import TextCompressProtocol
from transformers import AutoTokenizer
import re
import random
from .custom_dataset_loaders import load_custom_dataset
from typing import Iterator
import tqdm


class Challenger:
    def __init__(self):
        self.raw_datasets: list[IterableDataset] = self._load_raw_dataset()
        self.qa_datasets: list[IterableDataset] = self._load_qa_dataset()
        self.sat = "[START-ACTIVATE-TOKEN]"
        self.eat = "[END-ACTIVATE-TOKEN]"

    def __call__(
        self,
        tokenizer: AutoTokenizer,
        task: str = "",
        max_context_length_in_chars: int = 1536,
    ) -> TextCompressProtocol:
        r"""
        Get a sample for the given task.
        Args:
        - task (str): The task.
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        """
        if task == "qa":
            return self._get_qa_sample(tokenizer, max_context_length_in_chars)
        else:
            return self._get_ae_sample(tokenizer, max_context_length_in_chars)

    def _get_context(self, max_context_length_in_chars: int) -> str:
        r"""
        Get a context that can be used for tasks.
        Args:
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        """
        total_contexts_chars = 0
        contexts = []
        question = ""
        answer = ""
        qa_dataset = random.choice(self.qa_datasets)
        pbar = tqdm.tqdm(max_context_length_in_chars)
        for item in qa_dataset:
            context = item["context"]
            if total_contexts_chars + len(context) >= max_context_length_in_chars:
                break
            question = item["question"]
            answer = item["answer"]
            contexts.append(context)
            total_contexts_chars += len(context)
            pbar.write(f"contexts-char: {total_contexts_chars}")
            pbar.update(len(context))

        raw_dataset = random.choice(self.raw_datasets)
        for item in raw_dataset:
            context = item["text"]
            sentences = context.split(".")
            is_full = False
            for sentence in sentences:
                if total_contexts_chars + len(sentence) >= max_context_length_in_chars:
                    is_full = True
                    break
                contexts.append(sentence)
                total_contexts_chars += len(sentence)
                pbar.write(f"contexts-char: {total_contexts_chars}")
                pbar.update(len(sentence))
            if is_full:
                break
        pbar.close()
        print(
            f"Gathered {total_contexts_chars} characters for the context. Total contexts: {len(contexts)}"
        )
        return {
            "contexts": contexts,
            "question": question,
            "answer": answer,
        }

    def _get_qa_sample(
        self, tokenizer: AutoTokenizer, max_context_length_in_chars: int
    ) -> TextCompressProtocol:
        r"""
        Get a sample for the qa task.
        In this task, the assistant is asked to answer a question given the context.
        Args:
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        """
        item = self._get_context(max_context_length_in_chars)
        question = item["question"]
        answer = item["answer"]
        contexts = item["contexts"]

        prompt_after_context = f"\n\nAnswer the following question: {question}"

        contexts_str = "\n".join(contexts)

        messages = [
            {
                "role": "user",
                "content": f"""Given the context: {contexts_str}.{self.sat} {prompt_after_context}{self.eat}""",
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

    def _get_ae_sample(
        self, tokenizer: AutoTokenizer, max_context_length_in_chars: int
    ) -> TextCompressProtocol:
        r"""
        Get a sample for the autoencoder task.
        In this task, the assistant is asked to rewrite the same text given the context.
        Args:
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        - k (int): The length of the text to be compressed.
        """
        item = self._get_context(max_context_length_in_chars)
        contexts = item["contexts"]
        contexts_str = "\n".join(contexts)
        prompt_after_context = "\n\nRewrite exactly the same context."
        messages = [
            {
                "role": "user",
                "content": f"""Given the context: {contexts_str}.{self.sat} {prompt_after_context}{self.eat}""",
            },
            {
                "role": "assistant",
                "content": contexts_str,
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

    def _load_qa_dataset(self) -> list[IterableDataset]:
        r"""
        Combine multiple QA datasets.
        Return a combined dataset with the following schema:
        ------------------------------
        | contexts | question | answer |
        ------------------------------
        We filter with limited context length for easier to control in generating challenge.
        We can increase the number of datasets by adding more datasets.
        """
        datasets: list[IterableDataset] = [
            load_custom_dataset("qa_zre"),
        ]
        datasets = [
            ds.shuffle().filter(
                lambda x: len(x["context"]) < 512
                and self._is_mostly_alphabetic(x["context"])
            )
            for ds in datasets
        ]
        return datasets

    def _load_raw_dataset(self) -> Iterator[dict]:
        r"""
        Load the raw dataset for ae task.
        Raw dataset mean it only contains pieces of text.
        We can increase it by adding more datasets but for now, we only use FineWeb-pro (64 million paragraphs).
        """
        raw_datasets = [
            load_dataset("gair-prox/FineWeb-pro", streaming=True, split="train")
        ]
        raw_datasets = [
            ds.shuffle().filter(lambda x: self._is_mostly_alphabetic(x["text"]))
            for ds in raw_datasets
        ]
        return raw_datasets

    def _is_mostly_alphabetic(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if more than the specified threshold of characters in the text are alphabetic.
        :param text: The input text to check.
        :param threshold: The minimum proportion of alphabetic characters required.
        :return: True if the proportion of alphabetic characters is above the threshold, False otherwise.
        """
        alphabetic_count = sum(1 for char in text if char.isalpha())
        total_count = len(text)

        # Avoid division by zero
        if total_count == 0:
            return False

        return alphabetic_count / total_count > threshold
