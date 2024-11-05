import litserve as ls
from llmlingua import PromptCompressor
import transformers
import torch
from copy import deepcopy
import rich
import numpy as np


class InferenceLogger(ls.Logger):
    """
    Logger class for inference processes. This logs key-value pairs of information
    during the execution of the backend inference, using the rich library for better readability.
    """

    def process(self, key, value):
        # Logs the backend process with a specific key and value using rich.
        rich.print(f"Inference Backend -- {key}: {value}")


class Inference(ls.LitAPI):
    """
    Main inference class that implements methods for model setup, prediction, and encoding responses.
    This class interfaces with the LitAPI framework to handle incoming requests and return predictions.
    """

    def setup(self, device):
        """
        Setup the inference environment by loading the prompt compressor and model artifacts.

        Args:
            device (str): Specifies the device (e.g., CPU, GPU) on which the model will run.

        Initializes the `PromptCompressor` and loads the tokenizer and embeddings
        for the specified models.
        """
        # Initialize the prompt compressor with a specified model.
        self.llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map=device,
        )
        # Set the active models that will be used for inference.
        self.active_models = ["mistralai/Mistral-7B-Instruct-v0.1"]
        # Load model artifacts (tokenizer and embeddings) for active models.
        self.model_artifacts = {
            model: self._load_model_artifacts(model) for model in self.active_models
        }

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
        # Extract context and target model from the request.
        context = request["context"]
        target_model = request["target_model"]

        # Compress the prompt using the PromptCompressor with specified tokens and rate.
        compressed_context = self.llm_lingua.compress_prompt(
            context, rate=0.5, force_tokens=["\n", "?"]
        )

        # Log the number of tokens before and after compression.
        self.log(
            "Predict",
            f"Original tokens: {compressed_context['origin_tokens']} -> Compressed tokens: {compressed_context['compressed_tokens']}",
        )

        # Return the compressed context and the target model for the next step.
        return {
            "compressed_context": compressed_context["compressed_prompt"],
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
        # Extract compressed context and target model from the prediction.
        compressed_context = prediction["compressed_context"]
        target_model = prediction["target_model"]

        # Retrieve the tokenizer and token embeddings for the target model.
        tokenizer = self.model_artifacts[target_model]["tokenizer"]
        token_embeddings = self.model_artifacts[target_model]["token_embeddings"]

        # Tokenize the compressed context and obtain token IDs.
        compressed_ids = tokenizer(compressed_context, return_tensors="pt")["input_ids"]

        # Get the embeddings for the compressed token IDs.
        compressed_embeddings = token_embeddings(compressed_ids)

        # Log the shape of the resulting compressed embeddings.
        self.log("Encode Response", f"Compressed to {compressed_embeddings.shape}")
        compressed_embeddings = (
            np.array(compressed_embeddings.cpu()).astype(np.float32).tolist()
        )
        # Return the compressed token embeddings as a list.
        return {"compressed_tokens": compressed_embeddings}

    def _load_model_artifacts(self, model):
        """
        Helper function to load the model's tokenizer and embeddings.

        Args:
            model (str): The model name to load.

        Returns:
            dict: Contains tokenizer and input embeddings for the specified model.
        """
        # Load the pre-trained model and tokenizer using the HuggingFace transformers library.
        _model = transformers.AutoModelForCausalLM.from_pretrained(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)

        # Deep copy the model's input embeddings and delete the model instance to save memory.
        token_embeddings = deepcopy(_model.get_input_embeddings())
        del _model

        # Return the tokenizer and embeddings in a dictionary.
        return {
            "tokenizer": tokenizer,
            "token_embeddings": token_embeddings,
        }


if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing for configuring server settings.
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--devices", type=str, default="cpu")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--api-path", type=str, default="/condense")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--max-batch-size", type=int, default=1)
    args = parser.parse_args()

    # Instantiate the inference class and LitServer.
    inference = Inference()
    server = ls.LitServer(
        inference,
        devices=args.devices,
        workers_per_device=args.workers_per_device,
        timeout=args.timeout,
        api_path=args.api_path,
        accelerator=args.accelerator,
        max_batch_size=args.max_batch_size,
        loggers=[InferenceLogger()],
    )

    # Run the server on the specified port.
    server.run(port=args.port)
