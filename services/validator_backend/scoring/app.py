from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self.unit_test()

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
        self,
        request: BatchedScoringRequest,
        model,
        tokenizer,
    ):
        # Prepare the input sequence

        device = model.device

        context = request.ground_truth_request.context
        activation_prompt = request.ground_truth_request.activation_prompt
        expected_completion = request.ground_truth_request.expected_completion

        activation_prompt_tokens = tokenizer(
            activation_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        expected_completion_tokens = tokenizer(
            expected_completion, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        activation_prompt_embeddings = model.get_input_embeddings()(
            activation_prompt_tokens
        )

        expected_completion_embeddings = model.get_input_embeddings()(
            expected_completion_tokens
        )

        # Prepare loss list for miner responses
        losses = []

        for miner_response in request.miner_responses:
            compressed_tokens = (
                torch.tensor(miner_response.compressed_tokens).unsqueeze(0).to(device)
            )  # Shape (1, total_seq_len, hidden_size)
            inputs_embeddings = torch.cat(
                [
                    compressed_tokens,
                    activation_prompt_embeddings,
                    expected_completion_embeddings,
                ],
                dim=1,
            ).to(device)

            labels = torch.cat(
                [
                    torch.full(
                        (1, compressed_tokens.shape[1]), -100, dtype=torch.long
                    ).to(device),
                    activation_prompt_tokens,
                    expected_completion_tokens,
                ],
                dim=1,
            ).to(
                device
            )  # Shape (1, total_seq_len)

            labels = labels[:, 1:]  # Remove the first token from the labels

            outputs = model(
                inputs_embeds=inputs_embeddings,
            )
            logits = outputs.logits
            logits = logits[:, :-1, :]  # Remove the last token from the logits

            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
            )
            losses.append(loss.item())
        print("losses", losses)
        return losses

    def calculate_bleu_criteria(
        self, request: BatchedScoringRequest, model, tokenizer
    ) -> np.ndarray:
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
                inputs_embeds=inputs["inputs_embeds"],
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

    def unit_test(self):
        data = {
            "context": "<s> [INST] French senior civil servant arrested on suspicion of spying for North Korea\n\nNovember 27, 2018 by Joseph Fitsanakis\n\nA senior civil servant in the upper house of the French parliament has been arrested on suspicion of spying for North Korea, according to prosecutors. The news of the suspected spy\u2019s arrest was first reported on Monday by Quotidien, a daily politics and culture show on the Monaco-based television channel TMC. The show cited \u201ca judicial source in Paris\u201d and said that France\u2019s domestic security and counterintelligence agency, the General Directorate for Internal Security (DGSI), was in charge of the espionage case.\n\nThe senior administrator has been identified as Benoit Quennedey, a civil servant who liaises between the French Senate and the Department of Architecture and Heritage, which operates under France\u2019s Ministry of Culture. Quennedey was reportedly detained on Sunday morning and his office in the French Senate was raided by DGSI officers on the same day. Quotidien said that he was arrested on suspicion of \u201ccollecting and delivering to a foreign power information likely to subvert core national interests\u201d. The report did not provide specific information about the type of information that Quennedey is believed to have passed to North Korea. It did state, however, that a counterintelligence investigation into his activities began in March of this year.\n\nQuennedey is believed to be the president of the Franco-Korean Friendship Association, the French branch of a Spanish-based organization that lobbies in favor of international support for North Korea. Korea Friendship Association branches exist in over 30 countries and are believed to be officially sanctioned by Pyongyang. They operate as something akin to the pre-World War II Comintern (Communist International), a Moscow-sanctioned international pressure group that advocated in favor of Soviet-style communism around the world. French media reported on Monday that Quennedey traveled extensively to the Korean Peninsula in the past decade and has written a French-language book on North Korea. News reports said that the French President Emmanuel Macron had been made aware of Quennedey\u2019s arrest. The senior civil servant faces up to 30 years in prison if found guilty of espionage.\n\n\u25ba Author: Joseph Fitsanakis | Date: 27 November 2018 | Permalink\n\n",
            "activation_prompt": "Identify the person arrested on suspicion of spying for North Korea. [/INST]",
            "expected_completion": "Benoit Quennedey",
        }
        model_name = "Condense-AI/Mistral-7B-Instruct-v0.2"
        criterias = ["loss", "bleu"]

        # Create miner response fake from the model by converting context to embed_tokens
        context = data["context"]
        activation_prompt = data["activation_prompt"]
        expected_completion = data["expected_completion"]
        model_name = model_name
        criterias = criterias
        self.load_model(model_name)

        context_ids = self.tokenizers[model_name](
            context,
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        context_embeds = (
            self.models[model_name].get_input_embeddings()(context_ids).squeeze(0)
        )
        compressed_tokens = context_embeds.detach().cpu().numpy()
        print("unit-test", compressed_tokens.shape)
        compressed_tokens = compressed_tokens.tolist()
        miner_response = ScoringRequest(compressed_tokens=compressed_tokens)
        ground_truth_request = GroundTruthRequest(
            context=context,
            activation_prompt=activation_prompt,
            expected_completion=expected_completion,
            model_name=model_name,
            criterias=criterias,
        )
        request = BatchedScoringRequest(
            miner_responses=[miner_response], ground_truth_request=ground_truth_request
        )
        scores = self.get_scoring(request)

        print(scores)


app = FastAPI()
scoring_service = ScoringService()


@app.get("/")
def is_alive():
    return {"message": "I'm alive!"}


@app.post("/scoring")
def get_scoring(request: BatchedScoringRequest):
    return scoring_service.get_scoring(request)
