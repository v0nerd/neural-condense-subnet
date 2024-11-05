import torch
from transformers import HfArgumentParser
from peft import LoraConfig

from ICAE import ICAE, ModelArguments, AdditionalArguments
from safetensors.torch import load_file
import rich
import litserve as ls
import numpy as np
import base64
import io


def ndarray_to_base64(array: np.ndarray) -> str:
    """Convert a NumPy array to a base64-encoded string."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return base64_str


class InferenceLogger(ls.Logger):
    """
    Logger class for inference processes. This logs key-value pairs of information
    during the execution of the backend inference, using the rich library for better readability.
    """

    def process(self, key, value):
        rich.print(f"Inference Backend -- {key}: {value}")


class Inference(ls.LitAPI):
    def __init__(self, model_args, lora_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_args = model_args
        self.lora_config = lora_config

    def setup(self, device):
        self.device = device
        self.model_compress = ICAE(self.model_args, self.lora_config)
        state_dict = load_file(self.model_args.checkpoint_path)
        self.model_compress.load_state_dict(
            state_dict, strict=False
        )  # only load lora and memory token embeddings
        self.model_compress = self.model_compress.to(self.device)

    @torch.no_grad()
    def decode_request(self, request):
        """
        Decodes the incoming request for further processing.

        Args:
            request (dict): Incoming request data that includes context and other metadata.

        Returns:
            The original request as it doesn't modify the input.
        """
        return request

    @torch.no_grad()
    def predict(self, request):
        """
        Performs the prediction by compressing the input context and logging the compression details.

        Args:
            request (dict): Incoming request containing context and target model.

        Returns:
            dict: Compressed context and the target model name for further processing.
        """
        context = request["context"]
        target_model = request["target_model"]
        tokenized_input = self.model_compress.tokenizer(
            context,
            truncation=True,
            max_length=5120,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = torch.LongTensor([tokenized_input["input_ids"]]).to(self.device)
        memory_slots = self.model_compress._compress(input_ids)

        self.log(
            "Predict",
            f"Compress token length {len(memory_slots)} shape{memory_slots.shape}",
        )
        memory_slots = memory_slots.cpu().numpy()
        memory_slots = ndarray_to_base64(memory_slots)
        return {
            "compressed_tokens_b64": memory_slots,
            "target_model": target_model,
        }

    @torch.no_grad()
    def encode_response(self, prediction):
        """
        Encodes the compressed context into model-specific embeddings.

        Args:
            prediction (dict): Contains the compressed context and target model information.

        Returns:
            dict: Embeddings of the compressed tokens as a list.
        """

        return prediction


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, AdditionalArguments))
    model_args, additional_args = parser.parse_args_into_dataclasses()
    lora_config = LoraConfig(
        r=512,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    inference = Inference(
        model_args=model_args,
        lora_config=lora_config,
    )

    print("Inference backend started.")

    server = ls.LitServer(
        inference,
        devices=additional_args.devices,
        workers_per_device=additional_args.workers_per_device,
        timeout=additional_args.timeout,
        api_path=additional_args.api_path,
        accelerator=additional_args.accelerator,
        max_batch_size=additional_args.max_batch_size,
        loggers=[InferenceLogger()],
    )

    print("Inference backend server started.")

    server.run(port=additional_args.port)
