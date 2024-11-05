from datasets import load_dataset, IterableDataset
from ..protocol import TextCompressProtocol
from transformers import AutoTokenizer
import re
import random
from .custom_dataset_loaders import load_custom_dataset
from typing import Iterator, List, Dict
import tqdm


class Challenger:
    def __init__(self):
        self.raw_datasets: List[Iterator] = self._load_raw_dataset()
        self.qa_datasets: List[Iterator] = self._load_qa_dataset()
        self.conversation_datasets: List[Iterator] = self._load_conversation_dataset()
        self.sat = "[START-ACTIVATE-TOKEN]"
        self.eat = "[END-ACTIVATE-TOKEN]"

    def __call__(
        self,
        tokenizer: AutoTokenizer,
        task: str = "",
        max_context_length_in_chars: int = 1536,
    ) -> TextCompressProtocol:
        """
        Generate a sample challenge based on the specified task.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to be used.
            task (str, optional): The task type ('qa', 'ae', or other). Defaults to "".
            max_context_length_in_chars (int, optional): Maximum allowed context length in characters. Defaults to 1536.

        Returns:
            TextCompressProtocol: The protocol containing context, activation prompt, and expected completion.
        """
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
        Gather context up to the specified maximum character length.

        Args:
            max_context_length_in_chars (int): Maximum allowed context length in characters.

        Returns:
            Dict[str, str]: A dictionary containing 'contexts', 'question', and 'answer'.
        """
        total_context_chars = 0
        contexts = []
        question = ""
        answer = ""

        # Get QA context
        qa_dataset = random.choice(self.qa_datasets)
        total_context_chars = 0

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

        # Get raw text context
        raw_dataset = random.choice(self.raw_datasets)
        total_context_chars = 0

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

        return {
            "contexts": contexts,
            "question": question,
            "answer": answer,
        }

    def _get_qa_sample(
        self, tokenizer: AutoTokenizer, max_context_length_in_chars: int
    ) -> TextCompressProtocol:
        """
        Generate a QA sample challenge.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to be used.
            max_context_length_in_chars (int): Maximum allowed context length in characters.

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
        Generate an Autoencoder (AE) sample challenge.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to be used.
            max_context_length_in_chars (int): Maximum allowed context length in characters.

        Returns:
            TextCompressProtocol: The protocol for the AE challenge.
        """
        conversations, _, _ = self._get_conversations_mixed_with_context(
            max_context_length_in_chars
        )
        ae_target, format_of_a_turn = self._get_conversations_in_one_string(
            conversations
        )
        conversations[-1]["content"] += self.sat
        prompt_after_context = f"Rewrite whole conversation in a single string using the following format:\n{format_of_a_turn}. Each turn should be separated by a newline character."

        messages = conversations + [
            {"role": "user", "content": prompt_after_context},
            {"role": "assistant", "content": f"{self.eat}{ae_target}"},
        ]

        return self._build_protocol(tokenizer, messages)

    def _get_conversational_sample(
        self, tokenizer: AutoTokenizer, max_context_length_in_chars: int
    ) -> TextCompressProtocol:
        """
        Generate a conversational sample challenge where the assistant continues the conversation.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to be used.
            max_context_length_in_chars (int): Maximum allowed context length in characters.

        Returns:
            TextCompressProtocol: The protocol for the conversational challenge.
        """
        conversations = self._assemble_conversations(n_turns=random.choice([2, 3]))
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
        Get a mixed set of conversations and context.
        Context is gathered from QA and raw text datasets.
        Question & answer are relevant to the a piece of context.
        """
        item = self._get_context(max_context_length_in_chars)
        contexts_str = "\n".join(item["contexts"])
        question = item["question"]
        answer = item["answer"]

        conversations = self._assemble_conversations()
        # index_to_insert_context should be even to insert context after user message
        index_to_insert_context = random.choice(range(0, len(conversations), 2))
        conversations[index_to_insert_context]["content"] = (
            contexts_str + "\n" + conversations[index_to_insert_context]["content"]
        )

        return conversations, question, answer

    def _get_conversations_in_one_string(
        self, conversations: List[Dict[str, str]]
    ) -> str:
        """
        Convert a list of conversations into a single string.
        Format: "User: <User message>\nAssistant: <Assistant message>\nUser: <User message>..."
        Args:
            conversations (List[Dict[str, str]]): A list of conversation messages.

        Returns:
            str: The concatenated conversation string.
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
        self, n_turns: int = None, include_last_assistant: bool = True
    ) -> List[Dict[str, str]]:
        """
        Assemble a series of conversation turns.

        Args:
            n_turns (int, optional): Number of turns to include. Randomly chosen between 2 and 3 if not specified.
            include_last_assistant (bool, optional): Whether to include the last assistant turn. Defaults to True.

        Returns:
            List[Dict[str, str]]: A list of conversation messages.
        """
        if n_turns is None:
            n_turns = random.choice([2, 3])
        conversations = []

        for _ in range(n_turns):
            convo_dataset = random.choice(self.conversation_datasets)
            item = next(convo_dataset)
            conversations.extend(
                [
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]},
                ]
            )

        if not include_last_assistant:
            conversations = conversations[:-1]

        return conversations

    def _build_protocol(
        self,
        tokenizer: AutoTokenizer,
        messages: List[Dict[str, str]],
        expected_completion: str = None,
    ) -> TextCompressProtocol:
        """
        Build the TextCompressProtocol from messages.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to be used.
            messages (List[Dict[str, str]]): The list of conversation messages.
            expected_completion (str, optional): The expected completion text. If not provided, it will be extracted from messages.

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
        Load and prepare QA datasets.

        Returns:
            List[IterableDataset]: A list of QA datasets.
        """
        datasets = [load_custom_dataset("qa_zre")]
        return [iter(ds.shuffle()) for ds in datasets]

    def _load_raw_dataset(self) -> List[IterableDataset]:
        """
        Load and prepare raw text datasets.

        Returns:
            List[IterableDataset]: A list of raw text datasets.
        """
        datasets = [
            load_dataset("gair-prox/FineWeb-pro", streaming=True, split="train")
        ]
        return [iter(ds.shuffle()) for ds in datasets]

    def _load_conversation_dataset(self) -> List[IterableDataset]:
        """
        Load and prepare conversation datasets.

        Returns:
            List[IterableDataset]: A list of conversation datasets.
        """
        datasets = [load_custom_dataset("open_hermes")]
        return [iter(ds.shuffle()) for ds in datasets]
