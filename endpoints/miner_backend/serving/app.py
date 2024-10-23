import litserve as ls
from llmlingua import PromptCompressor
import transformers
import torch


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map=device,
        )
        self.active_models = ["meta-llama/Llama-3.1-8B-Instruct"]
        self.model_artifacts = {
            model: self._load_model_artifacts(model) for model in self.active_models
        }

    def decode_request(self, request):
        context = request["context"]
        target_model = request["target_model"]
        return context, target_model

    def predict(self, context, target_model):
        compressed_context = self.llm_lingua.compress_prompt(
            context, rate=0.33, force_tokens=["\n", "?"]
        )
        return compressed_context, target_model

    def encode_response(self, compressed_context, target_model):
        tokenizer = self.model_artifacts[target_model]["tokenizer"]
        token_embeddings = self.model_artifacts[target_model]["token_embeddings"]
        compressed_ids = tokenizer(compressed_context, return_tensors="pt")
        compressed_embeddings = token_embeddings(compressed_ids.input_ids)
        return {"compressed_tokens": compressed_embeddings}

    def _load_model_artifacts(self, model):
        _model = transformers.AutoModelForCausalLM.from_pretrained(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        token_embeddings = torch.clone(_model.get_input_embeddings())
        del _model
        return {
            "tokenizer": tokenizer,
            "token_embeddings": token_embeddings,
        }


if __name__ == "__main__":
    server = ls.LitServer(SimpleLitAPI(), accelerator="auto", max_batch_size=1)
    server.run(port=8080)
