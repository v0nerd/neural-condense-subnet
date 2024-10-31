from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import gc
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextGenerationPipeline
from .utils import loss_to_scores, calculate_bleu
import threading


class ScoringRequest(BaseModel):
    compressed_tokens: List[List[float]]


class GroundTruthRequest(BaseModel):
    context: str
    expected_completion: str
    activation_prompt: str
    model_name: str
    criterias: List[str]


class BatchedScoringRequest(BaseModel):
    miner_responses: List[ScoringRequest]
    ground_truth_request: GroundTruthRequest


class ScoringService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name: str):
        with self.lock:
            if model_name not in self.models:
                self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name
                )
                self.models[model_name].to(self.device)
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def get_scoring(self, request: BatchedScoringRequest):
        model_name = request.ground_truth_request.model_name
        self.load_model(model_name)
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        outputs = []

        if "loss" in request.ground_truth_request.criterias:
            scores = self.calculate_loss_criteria(request, model, tokenizer)
            outputs.append(scores)

        if "bleu" in request.ground_truth_request.criterias:
            scores = self.calculate_bleu_criteria(request, model, tokenizer)
            outputs.append(scores)

        scores = np.mean(outputs, axis=0)
        return {
            "scores": scores.tolist()
        }  # Convert numpy array to list for JSON serialization

    def calculate_loss_criteria(
        self, request: BatchedScoringRequest, model, tokenizer
    ) -> np.ndarray:
        original_labels = (
            tokenizer(
                request.ground_truth_request.expected_completion,
                return_tensors="pt",
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            .squeeze(0)
            .to(model.device)
        )
        context = request.ground_truth_request.context
        activation_prompt = request.ground_truth_request.activation_prompt
        losses = []

        for miner_output in request.miner_responses:
            n_compressed_tokens = len(miner_output.compressed_tokens)
            prefix_labels = torch.full(
                (n_compressed_tokens,), -52, dtype=torch.long
            ).to(model.device)
            labels = torch.cat([prefix_labels, original_labels])
            labels = labels.unsqueeze(0).to(model.device)
            labels = labels[:, 1:].reshape(-1)
            activation_input_ids = tokenizer(
                activation_prompt,
                return_tensors="pt",
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"].to(model.device)
            inputs = self.prepare_condensed_inputs(
                miner_output.compressed_tokens,
                activation_input_ids,
                model.get_input_embeddings(),
            )
            outputs = model(**inputs)
            logits = outputs.logits
            effective_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            loss = F.cross_entropy(effective_logits, labels, ignore_index=-52)
            losses.append(loss.item())
        scores = loss_to_scores(losses)
        return scores

    def calculate_bleu_criteria(
        self, request: BatchedScoringRequest, model, tokenizer
    ) -> np.ndarray:
        context = request.ground_truth_request.context
        activation_prompt = request.ground_truth_request.activation_prompt
        bleu_scores = []

        for miner_output in request.miner_responses:
            activation_input_ids = tokenizer(
                activation_prompt,
                return_tensors="pt",
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"].to(model.device)
            inputs = self.prepare_condensed_inputs(
                miner_output.compressed_tokens,
                activation_input_ids,
                model.get_input_embeddings(),
            )
            generated_outputs = model.generate(
                inputs_embeds=inputs["input_embeds"],
                max_length=64,
                num_return_sequences=1,
            )
            completion = tokenizer.decode(
                generated_outputs[0], skip_special_tokens=True
            )
            bleu_score = calculate_bleu(
                request.ground_truth_request.expected_completion, completion
            )
            bleu_scores.append(bleu_score)

        bleu_scores = np.array(bleu_scores)
        # Normalize BLEU scores if sum is not zero
        total = np.sum(bleu_scores)
        if total > 0:
            bleu_scores = bleu_scores / total
        else:
            bleu_scores = np.zeros_like(bleu_scores)
        return bleu_scores

    def prepare_condensed_inputs(
        self,
        condensed_tokens: List[List[float]],
        input_ids: List[int],
        embed_tokens: torch.nn.Embedding,
        device="cuda",
    ):
        r"""
        Prepare the inputs for the model.
        Args:
        - condensed_tokens (List[List[float]]): The condensed tokens.
        - input_ids (List[int]): The input ids to be concatenated with the condensed tokens.
        Returns:
        - inputs (dict): The inputs for the model.
        """
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        else:
            assert isinstance(input_ids, torch.Tensor)
        if isinstance(condensed_tokens, list):
            condensed_tokens = torch.FloatTensor(condensed_tokens).unsqueeze(0)
        else:
            assert isinstance(condensed_tokens, torch.Tensor)
        if condensed_tokens.dim() == 2:
            condensed_tokens = condensed_tokens.unsqueeze(0)

        input_tokens = embed_tokens(input_ids)
        condensed_tokens = condensed_tokens.to(input_tokens.device)
        print(condensed_tokens.shape, input_tokens.shape)
        inputs_embeds = torch.cat([condensed_tokens, input_tokens], dim=1)
        inputs_embeds = inputs_embeds.to(device)
        return {"inputs_embeds": inputs_embeds}


app = FastAPI()
scoring_service = ScoringService()


@app.get("/")
def is_alive():
    return {"message": "I'm alive!"}


@app.post("/scoring")
def get_scoring(request: BatchedScoringRequest):
    return scoring_service.get_scoring(request)
