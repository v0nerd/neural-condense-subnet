from bittensor import Synapse
from typing import List, Any
from .common import base64


class Metadata(Synapse):
    metadata: dict = {}


class TextCompressProtocol(Synapse):
    r"""
    Protocol for the Text-Compress task.
    Attributes:
    - Seen by miners
        - context (str): The text to be compressed by miners.
        - prompt_after_context (str): The postfix text to be concatenated with the text_to_compress.
    - Output of miners
        - compressed_tokens (List[List[float]]): The compressed tokens generated by the miner.
    - Ground truth seen by validators
        - expected_generation (str): The original generation.
    """

    context: str = ""
    compressed_tokens_b64: str = ""
    compressed_tokens: Any = []
    expected_completion: str = ""
    activation_prompt: str = ""
    target_model: str = ""

    last_prompt: str = ""

    def get_miner_payload(self):
        r"""
        Get the input for the miner.
        Returns:
        - miner_payload (dict): The input for the miner.
        """
        return {
            "context": self.context,
            "target_model": self.target_model,
        }

    def hide_ground_truth(self):
        r"""
        Hide the ground truth from the miner.
        """
        self.expected_completion = ""
        self.activation_prompt = ""
        self.last_prompt = ""

    def deserialize(self) -> Synapse:
        return {
            "context": self.context,
            "compressed_tokens_b64": self.compressed_tokens_b64,
            "expected_completion": self.expected_completion,
            "activation_prompt": self.activation_prompt,
            "last_prompt": self.last_prompt,
        }

    def base64_to_ndarray(self):
        r"""
        Convert the base64 string to np.ndarray.
        """
        self.compressed_tokens = base64.base64_to_ndarray(self.compressed_tokens_b64)
