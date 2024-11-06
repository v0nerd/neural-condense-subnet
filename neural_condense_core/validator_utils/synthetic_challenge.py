from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
import re
import random
from typing import Iterator, List, Dict
import threading
from .custom_dataset_loaders import load_custom_dataset
from ..protocol import TextCompressProtocol


class Challenger:
    def __init__(self):
        """
        Initialize the Challenger class with various dataset loaders and configuration tokens.
        """
        self.raw_datasets: List[Iterator] = self._load_raw_dataset()
        self.qa_datasets: List[Iterator] = self._load_qa_dataset()
        self.conversation_datasets: List[Iterator] = self._load_conversation_dataset()
        self.sat = "[START-ACTIVATE-TOKEN]"
        self.eat = "[END-ACTIVATE-TOKEN]"
        self.lock = threading.Lock()  # Ensures thread safety for dataset access

    def __call__(
        self,
        tokenizer: AutoTokenizer,
        task: str = "",
        max_context_length_in_chars: int = 1536,
    ) -> TextCompressProtocol:
        """
        Generates a sample challenge based on the specified task type.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for formatting the text.
            task (str, optional): The type of task ('question_answering', 'conversation', 'reconstruction'). Defaults to "".
            max_context_length_in_chars (int, optional): Max length of context in characters. Defaults to 1536.

        Returns:
            TextCompressProtocol: An object containing context, activation prompt, and expected completion.
        """
        with self.lock:
            if task == "question_answering":
                return self._get_qa_sample(tokenizer, max_context_length_in_chars)
            elif task == "conversation":
                return self._get_conversational_sample(
                    tokenizer, max_context_length_in_chars
                )
            elif task == "reconstruction":
                return self._get_ae_sample(tokenizer, max_context_length_in_chars)
            else:
                raise ValueError(f"Invalid task type: {task}")

    def _get_context(self, max_context_length_in_chars: int) -> Dict[str, str]:
        """
        Gather a dictionary containing context, a question, and an answer, limited by character length.

        Args:
            max_context_length_in_chars (int): Character limit for context.

        Returns:
            Dict[str, str]: Dictionary containing 'contexts', 'question', and 'answer'.
        """
        total_context_chars = 0
        contexts = []
        question, answer = "", ""

        # Retrieve context from the QA dataset
        qa_dataset = random.choice(self.qa_datasets)
        while True:
            try:
                item = next(qa_dataset)
            except StopIteration:
                break

            context = item["context"]
            if total_context_chars + len(context) >= max_context_length_in_chars:
                break

            question = item["question"]
            answer = item["answer"]
            contexts.append(context)
            total_context_chars += len(context)

        # Append additional context from raw text datasets
        raw_dataset = random.choice(self.raw_datasets)
        while True:
            try:
                item = next(raw_dataset)
            except StopIteration:
                break

            text = item["text"]
            sentences = text.split(".")

            for sentence in sentences:
                if total_context_chars + len(sentence) >= max_context_length_in_chars:
                    break
                contexts.append(sentence)
                total_context_chars += len(sentence)
            else:
                continue
            break

        return {"contexts": contexts, "question": question, "answer": answer}

    def _get_qa_sample(
        self, tokenizer: AutoTokenizer, max_context_length_in_chars: int
    ) -> TextCompressProtocol:
        """
        Generate a sample challenge for the Question-Answering (QA) task.

        Args:
            tokenizer (AutoTokenizer): Tokenizer to format the text.
            max_context_length_in_chars (int): Max allowed context length in characters.

        Returns:
            TextCompressProtocol: The protocol for the QA challenge.
        """
        conversations, question, answer = self._get_conversations_mixed_with_context(
            max_context_length_in_chars
        )
        conversations[-1]["content"] += self.sat
        prompt_after_context = f"\n\nAnswer the following question: {question}"

        messages = conversations + [
            {"role": "user", "content": prompt_after_context},
            {"role": "assistant", "content": f"{self.eat}{answer}"},
        ]

        return self._build_protocol(tokenizer, messages)

    def _get_ae_sample(
        self, tokenizer: AutoTokenizer, max_context_length_in_chars: int
    ) -> TextCompressProtocol:
        """
        Generate a sample challenge for the Autoencoder (AE) task.

        Args:
            tokenizer (AutoTokenizer): Tokenizer to format the text.
            max_context_length_in_chars (int): Max allowed context length in characters.

        Returns:
            TextCompressProtocol: The protocol for the AE challenge.
        """
        conversations, _, _ = self._get_conversations_mixed_with_context(
            max_context_length_in_chars // 2
        )
        ae_target, format_of_a_turn = self._get_conversations_in_one_string(
            conversations
        )

        conversations[-1]["content"] += self.sat
        prompt_after_context = f"Rewrite the entire conversation using the following format:\n{format_of_a_turn}. Each turn should be separated by a newline."

        messages = conversations + [
            {"role": "user", "content": prompt_after_context},
            {"role": "assistant", "content": f"{self.eat}{ae_target}"},
        ]

        return self._build_protocol(tokenizer, messages)

    def _get_conversational_sample(
        self, tokenizer: AutoTokenizer, max_context_length_in_chars: int
    ) -> TextCompressProtocol:
        """
        Generate a sample challenge where the assistant continues a conversation.

        Args:
            tokenizer (AutoTokenizer): Tokenizer to format the text.
            max_context_length_in_chars (int): Max allowed context length in characters.

        Returns:
            TextCompressProtocol: The protocol for the conversational challenge.
        """
        conversations = self._assemble_conversations(max_context_length_in_chars)
        expected_completion = conversations.pop()["content"]
        prompt_after_context = conversations.pop()["content"]

        conversations[-1]["content"] += self.sat
        messages = conversations + [
            {"role": "user", "content": prompt_after_context},
            {"role": "assistant", "content": f"{self.eat}{expected_completion}"},
        ]

        protocol = self._build_protocol(tokenizer, messages)
        protocol.last_prompt = prompt_after_context
        return protocol

    def _get_conversations_mixed_with_context(self, max_context_length_in_chars: int):
        """
        Retrieve a mixed set of context and conversations.

        Args:
            max_context_length_in_chars (int): Max allowed context length in characters.

        Returns:
            Tuple[List[Dict[str, str]], str, str]: Conversations, question, and answer.
        """
        item = self._get_context(max_context_length_in_chars // 2)
        contexts_str = "\n".join(item["contexts"])
        question, answer = item["question"], item["answer"]

        conversations = self._assemble_conversations(max_context_length_in_chars // 2)
        index_to_insert_context = random.choice(range(0, len(conversations), 2))
        conversations[index_to_insert_context]["content"] = (
            contexts_str + "\n" + conversations[index_to_insert_context]["content"]
        )

        return conversations, question, answer

    def _get_conversations_in_one_string(
        self, conversations: List[Dict[str, str]]
    ) -> str:
        """
        Convert a list of conversations to a single formatted string.

        Args:
            conversations (List[Dict[str, str]]): List of conversation messages.

        Returns:
            str: Formatted conversation string.
        """
        format_of_a_turn = "User: {user}\nAssistant: {assistant}"
        conversation_as_strings = []

        for i in range(1, len(conversations), 2):
            user_message = conversations[i - 1]["content"]
            assistant_message = conversations[i]["content"]
            conversation_as_strings.append(
                format_of_a_turn.format(user=user_message, assistant=assistant_message)
            )

        return "\n".join(conversation_as_strings), format_of_a_turn

    def _assemble_conversations(
        self, max_context_length_in_chars: int, include_last_assistant: bool = True
    ) -> List[Dict[str, str]]:
        """
        Collect a series of conversation turns up to a specified character limit.

        Args:
            max_context_length_in_chars (int): Max length of the conversation in characters.
            include_last_assistant (bool, optional): Include last assistant turn. Defaults to True.

        Returns:
            List[Dict[str, str]]: List of conversation messages.
        """
        conversations = []
        current_length = 0

        def fetch_valid_turn(dataset) -> Dict[str, str]:
            """Fetch a valid conversation turn within the character limit from the dataset."""
            for _ in range(5):
                try:
                    item = next(dataset)
                    turn_length = len(item["user"]) + len(item["assistant"])
                    if current_length + turn_length <= max_context_length_in_chars:
                        return item
                except StopIteration:
                    break
            return None

        for _ in range(10):
            convo_dataset = random.choice(self.conversation_datasets)
            item = fetch_valid_turn(convo_dataset)
            if item:
                current_length += len(item["user"]) + len(item["assistant"])
                conversations.extend(
                    [
                        {"role": "user", "content": item["user"]},
                        {"role": "assistant", "content": item["assistant"]},
                    ]
                )

        if not include_last_assistant and conversations:
            conversations.pop()

        return conversations

    def _build_protocol(
        self,
        tokenizer: AutoTokenizer,
        messages: List[Dict[str, str]],
        expected_completion: str = None,
    ) -> TextCompressProtocol:
        """
        Build the TextCompressProtocol from the list of messages.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for formatting.
            messages (List[Dict[str, str]]): List of conversation messages.
            expected_completion (str, optional): Expected response text. Defaults to None.

        Returns:
            TextCompressProtocol: The constructed protocol.
        """
        messages_str = tokenizer.apply_chat_template(messages, tokenize=False)
        parts = re.split(f"{re.escape(self.sat)}|{re.escape(self.eat)}", messages_str)
        context, activation_prompt, expected_completion = parts

        return TextCompressProtocol(
            context=context.strip(),
            activation_prompt=activation_prompt.strip(),
            expected_completion=expected_completion.strip(),
        )

    def _load_qa_dataset(self) -> List[IterableDataset]:
        """
        Load and shuffle the QA datasets.

        Returns:
            List[IterableDataset]: Shuffled QA datasets.
        """
        datasets = [load_custom_dataset("qa_zre")]
        return [InfiniteDataset(ds.shuffle()) for ds in datasets]

    def _load_raw_dataset(self) -> List[IterableDataset]:
        """
        Load and shuffle raw text datasets.

        Returns:
            List[IterableDataset]: Shuffled raw text datasets.
        """
        datasets = [
            load_dataset("gair-prox/FineWeb-pro", streaming=True, split="train")
        ]
        return [InfiniteDataset(ds.shuffle()) for ds in datasets]

    def _load_conversation_dataset(self) -> List[IterableDataset]:
        """
        Load and shuffle conversation datasets.

        Returns:
            List[IterableDataset]: Shuffled conversation datasets.
        """
        datasets = [load_custom_dataset("open_hermes")]
        return [InfiniteDataset(ds.shuffle()) for ds in datasets]


class InfiniteDataset:
    def __init__(self, dataset: IterableDataset):
        """
        Initialize the InfiniteDataset wrapper.

        Args:
            dataset (IterableDataset): The dataset object to wrap. It should be an iterable dataset.
        """
        self.dataset = dataset
        self.iterator = iter(self.dataset)  # Initialize the iterator

    def __iter__(self) -> Iterator:
        """
        Return the iterator for the dataset.

        Returns:
            Iterator: An iterator over the dataset.
        """
        return self

    def __next__(self):
        """
        Get the next item in the dataset. Automatically reinitialize the iterator if the end is reached.

        Returns:
            The next item in the dataset.
        """
        try:
            return next(self.iterator)
        except StopIteration:
            # Reinitialize iterator if the end is reached
            self.iterator = iter(self.dataset)
            return next(self.iterator)  # Return the first item of the new iterator
