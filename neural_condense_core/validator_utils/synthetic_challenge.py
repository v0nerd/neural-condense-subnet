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
                "content": f"""Given the context: {contexts_as_bullets}. {prompt_after_context}""",
            },
        ]
        context = self._get_context(messages[0], tokenizer, prompt_after_context)
        messages.append(
            {
                "role": "assistant",
                "content": answer,
            },
        )
        expected_completion = self._get_expected_completion(
            messages, tokenizer, context
        )

        assert context + expected_completion == tokenizer.apply_chat_template(
            messages, tokenize=False
        ), "The context and the expected completion do not match the messages."

        protocol = TextCompressProtocol(
            context=context,
            expected_completion=expected_completion,
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
                "content": f"""Given the context: {text}. {prompt_after_context}""",
            },
        ]
        context = self._get_context(messages[0], tokenizer, prompt_after_context)
        messages.append(
            {
                "role": "assistant",
                "content": text,
            },
        )
        expected_completion = self._get_expected_completion(
            messages, tokenizer, context
        )

        assert context + expected_completion == tokenizer.apply_chat_template(
            messages, tokenize=False
        ), "The context and the expected completion do not match the messages."

        protocol = TextCompressProtocol(
            context=context,
            expected_completion=expected_completion,
        )
        return protocol

    def _get_context(
        self, message: dict, tokenizer: AutoTokenizer, prompt_after_context: str
    ) -> str:
        r"""
        Get the context from the message.
        Args:
        - message (dict): The message.
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        """
        messages = [
            {
                "role": "user",
                "content": message["content"],
            },
        ]
        messages_str: str = tokenizer.apply_chat_template(messages, tokenize=False)
        # Remove all text after the prompt_after_context.
        context = re.sub(
            f"{re.escape(prompt_after_context)}.*", "", messages_str, flags=re.DOTALL
        )
        return context

    def _get_expected_completion(
        self, messages: list[dict], tokenizer: AutoTokenizer, context: str
    ) -> str:
        r"""
        Get the expected generation from the messages.
        Args:
        - messages (list[dict]): The messages.
        - tokenizer (AutoTokenizer): The tokenizer to be used.
        """
        messages_str: str = tokenizer.apply_chat_template(messages, tokenize=False)
        expected_generation = re.sub(
            f"{re.escape(context)}", "", messages_str, flags=re.DOTALL
        )
        return expected_generation

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
