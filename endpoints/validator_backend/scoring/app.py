from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import gc
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoTokenizer, TextGenerationPipeline
from .condensible_model_for_causal_lm import CondensibleModelForCausalLM
from .utils import loss_to_scores, calculate_bleu


class ScoringRequest(BaseModel):
    compressed_tokens: List[List[float]]


class GroundTruthRequest(BaseModel):
    context: str
    expected_completion: str
    model_name: str
    max_condensed_tokens: int
    criterias: List[str]


class BatchedScoringRequest(BaseModel):
    miner_responses: List[ScoringRequest]
    ground_truth_request: GroundTruthRequest


app = FastAPI()
MODEL: CondensibleModelForCausalLM = None
PIPE: TextGenerationPipeline = None
TOKENIZER: AutoTokenizer = None
CURRENT_MODEL_NAME: str = None


@app.get("/")
def is_alive():
    return {"message": "I'm alive!"}


@app.post("/scoring")
@torch.no_grad()
def get_scoring(request: BatchedScoringRequest):
    model_name = request.ground_truth_request.model_name
    if not MODEL:
        MODEL = CondensibleModelForCausalLM.from_pretrained(model_name)
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        CURRENT_MODEL_NAME = model_name

    if CURRENT_MODEL_NAME != model_name:
        del MODEL
        del TOKENIZER
        gc.collect()
        torch.cuda.empty_cache()
        MODEL = CondensibleModelForCausalLM.from_pretrained(model_name)
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        CURRENT_MODEL_NAME = model_name

    outputs = []
    if "loss" in request.ground_truth_request.criterias:
        scores = calculate_loss_criteria(request)
        outputs.append(scores)

    if "bleu" in request.ground_truth_request.criterias:
        scores = calculate_bleu_criteria(request)
        outputs.append(scores)

    scores = np.mean(outputs, axis=0)
    return {"scores": scores}


def calculate_loss_criteria(request: BatchedScoringRequest) -> np.ndarray:
    original_labels = TOKENIZER.encode(
        request.expected_completion,
        return_tensors="pt",
        truncation=False,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False,
    )["input_ids"]
    context = request.ground_truth_request.context

    losses = []

    for miner_output in request.miner_responses:
        n_compressed_tokens = len(miner_output.compressed_tokens)
        labels = [-52] * n_compressed_tokens + original_labels
        labels = torch.LongTensor(labels).unsqueeze(0).to(MODEL.device)
        labels = labels[:, 1:].reshape(-1)
        n_compressed_tokens = len(miner_output.compressed_tokens)
        input_ids = TOKENIZER.encode(
            context,
            return_tensors="pt",
            truncation=False,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        inputs = MODEL.prepare_condensed_inputs(
            miner_output.compressed_tokens, input_ids
        )
        outputs = MODEL(**inputs)
        logits = outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        loss = F.cross_entropy(effective_logits, labels, ignore_index=-52)
        losses.append(loss.item())
    scores = loss_to_scores(losses)

    return scores


def calculate_bleu_criteria(request: BatchedScoringRequest) -> np.ndarray:
    PIPE = TextGenerationPipeline(MODEL, TOKENIZER)
    context = request.ground_truth_request.context
    bleu_scores = []

    for miner_output in request.miner_responses:
        completions = PIPE(
            context, max_length=64, condensed_tokens=miner_output.compressed_tokens
        )
        completion = completions[0]["generated_text"]
        bleu_score = calculate_bleu(
            request.ground_truth_request.expected_completion, completion
        )
        bleu_scores.append(bleu_score)

    bleu_scores = np.array(bleu_scores)
    bleu_scores = bleu_scores / np.sum(bleu_scores)
    return bleu_scores


def calculate_compress_rate(request: BatchedScoringRequest) -> np.ndarray:
    context = request.ground_truth_request.context
    compress_rates = []

    for miner_output in request.miner_responses:
        n_compressed_tokens = len(miner_output.compressed_tokens)
        input_ids = TOKENIZER.encode(
            context,
            return_tensors="pt",
            truncation=False,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        compress_rate = len(input_ids) / n_compressed_tokens
        compress_rates.append(compress_rate)

    compress_rates = np.array(compress_rates)
    compress_rates = compress_rates / np.sum(compress_rates)
    return compress_rates
