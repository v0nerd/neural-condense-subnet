from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import torch.nn.functional as F
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from .utils import loss_to_scores
import threading
import traceback
import logging
import io
import base64

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Validator-Backend")


def base64_to_ndarray(base64_str: str) -> np.ndarray:
    try:
        """Convert a base64-encoded string back to a NumPy array."""
        buffer = io.BytesIO(base64.b64decode(base64_str))
        buffer.seek(0)
        array = np.load(buffer)
        array = array.astype(np.float32)
    except Exception as e:
        print(e)
        return None
    return array


def ndarray_to_base64(array: np.ndarray) -> str:
    try:
        """Convert a NumPy array to a base64-encoded string."""
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print(e)
        return ""
    return base64_str


class ScoringRequest(BaseModel):
    compressed_tokens_b64: str
    compressed_tokens: Any = None


class GroundTruthRequest(BaseModel):
    context: str
    expected_completion: str
    activation_prompt: str
    model_name: str
    criterias: List[str]
    last_prompt: str = ""


class BatchedScoringRequest(BaseModel):
    miner_responses: List[ScoringRequest]
    ground_truth_request: GroundTruthRequest


class ScoringService:
    def __init__(self):
        """
        Initializes the ScoringService with model and tokenizer storage, device configuration,
        and a lock for thread-safe operations. Runs a unit test to verify setup.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reward_model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.models = {}
        self.tokenizers = {}
        self.lock = threading.Lock()
        self.unit_test()

    def load_model(self, model_name: str):
        """
        Loads a specified model and tokenizer if not already loaded, ensuring thread safety.
        """
        with self.lock:
            try:
                if model_name not in self.models:
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name
                    )
                    self.models[model_name].to(self.device)
                    self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                        model_name
                    )
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    @torch.no_grad()
    def get_scoring(self, request: BatchedScoringRequest):
        """
        Returns scoring based on criteria specified in the request, such as loss and accuracy,
        calculated from the given model and tokenizer.
        """
        try:
            model_name = request.ground_truth_request.model_name
            self.load_model(model_name)
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            outputs = []

            for miner_response in request.miner_responses:
                miner_response.compressed_tokens = base64_to_ndarray(
                    miner_response.compressed_tokens_b64
                )

            if "loss" in request.ground_truth_request.criterias:
                scores = self.calculate_loss_criteria(request, model, tokenizer)
                scores = self._smooth_scores(scores, delta_0=0.3, decay=0.5)
                outputs.append(scores)

            if "accuracy" in request.ground_truth_request.criterias:
                scores = self.calculate_accuracy_criteria(request, model, tokenizer)
                scores = self._smooth_scores(scores, delta_0=0.3, decay=0.5)
                outputs.append(scores)

            if "reward_model" in request.ground_truth_request.criterias:
                scores = self.calculate_llm_reward_criteria(request, model, tokenizer)
                scores = self._smooth_scores(scores, delta_0=0.3, decay=0.5)
                outputs.append(scores)

            scores = np.mean(outputs, axis=0)
            return {"scores": scores.tolist()}
        except Exception as e:
            traceback.print_exc()
            print(f"Error in get_scoring: {e}")
            return {"scores": []}

    def calculate_loss_criteria(self, request: BatchedScoringRequest, model, tokenizer):
        """
        Calculates the loss-based scores by comparing expected and generated tokens based
        on the activation prompt and expected completion.
        """
        device = model.device
        activation_prompt = request.ground_truth_request.activation_prompt
        expected_completion = request.ground_truth_request.expected_completion
        losses = []

        try:
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

            for miner_response in request.miner_responses:
                try:
                    compressed_tokens = (
                        torch.tensor(miner_response.compressed_tokens)
                        .unsqueeze(0)
                        .to(device)
                    )
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
                    ).to(device)

                    labels = labels[:, 1:]
                    outputs = model(inputs_embeds=inputs_embeddings)
                    logits = outputs.logits[:, :-1, :]

                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                    losses.append(loss.item())
                except Exception as e:
                    print(f"Error in calculate_loss_criteria loop: {e}")
                    losses.append(1000)
            scores = loss_to_scores(losses)
            logger.info(f"Losses: {losses}")
            return scores
        except Exception as e:
            print(f"Error in calculate_loss_criteria: {e}")
            return []

    def calculate_llm_reward_criteria(
        self, request: BatchedScoringRequest, model, tokenizer
    ):
        device = model.device
        activation_prompt = request.ground_truth_request.activation_prompt
        expected_completion = request.ground_truth_request.expected_completion
        last_prompt = request.ground_truth_request.last_prompt
        rewards = []

        activation_prompt_tokens = tokenizer(
            activation_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        activation_prompt_embeddings = model.get_input_embeddings()(
            activation_prompt_tokens
        )

        for miner_output in request.miner_responses:
            try:
                compressed_tokens = (
                    torch.tensor(miner_output.compressed_tokens).unsqueeze(0).to(device)
                )
                inputs_embeds = torch.cat(
                    [compressed_tokens, activation_prompt_embeddings], dim=1
                ).to(device)

                generated_outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=256,
                    num_return_sequences=1,
                )
                completion = tokenizer.decode(
                    generated_outputs[0], skip_special_tokens=True
                ).strip()

                conversation_to_score = [
                    {
                        "role": "user",
                        "content": last_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": completion,
                    },
                ]
                logger.info(f"Conversation Reward: {conversation_to_score}")
                conversation_to_score = self.rm_tokenizer.apply_chat_template(
                    conversation_to_score, tokenize=True, return_tensors="pt"
                ).to(device)
                score = self.rm_model(conversation_to_score).logits[0][0].item()
                logger.info(f"Reward Score: {score}")
                rewards.append(score)
            except Exception as e:
                traceback.print_exc()
                print(f"Error in calculate_accuracy_criteria loop: {e}")
                rewards.append(0)
        return rewards

    def calculate_accuracy_criteria(
        self, request: BatchedScoringRequest, model, tokenizer
    ):
        """
        Calculates accuracy-based scores by generating responses from miner responses,
        comparing them to the expected completion.
        """
        device = model.device
        activation_prompt = request.ground_truth_request.activation_prompt
        expected_completion = request.ground_truth_request.expected_completion
        accuracy_scores = []

        try:
            activation_prompt_tokens = tokenizer(
                activation_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            activation_prompt_embeddings = model.get_input_embeddings()(
                activation_prompt_tokens
            )

            for miner_output in request.miner_responses:
                try:
                    compressed_tokens = (
                        torch.tensor(miner_output.compressed_tokens)
                        .unsqueeze(0)
                        .to(device)
                    )
                    inputs_embeds = torch.cat(
                        [compressed_tokens, activation_prompt_embeddings], dim=1
                    ).to(device)

                    generated_outputs = model.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=64,
                        num_return_sequences=1,
                    )
                    completion = tokenizer.decode(
                        generated_outputs[0], skip_special_tokens=True
                    ).strip()
                    accuracy = self._llm_judge(
                        expected_completion, completion, model, tokenizer
                    )
                    accuracy_scores.append(accuracy)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error in calculate_accuracy_criteria loop: {e}")
                    accuracy_scores.append(0)
            return accuracy_scores
        except Exception as e:
            traceback.print_exc()
            print(f"Error in calculate_accuracy_criteria: {e}")
            return []

    def unit_test(self):
        """
        Runs a basic unit test to verify the setup and scoring functions for a sample request.
        """
        try:
            data = {
                "context": "<s> [INST] Provided the context: French senior civil servant arrested on suspicion of spying for North Korea\n\nNovember 27, 2018 by Joseph Fitsanakis\n\nA senior civil servant in the upper house of the French parliament has been arrested on suspicion of spying for North Korea, according to prosecutors. The news of the suspected spy\u2019s arrest was first reported on Monday by Quotidien, a daily politics and culture show on the Monaco-based television channel TMC. The show cited \u201ca judicial source in Paris\u201d and said that France\u2019s domestic security and counterintelligence agency, the General Directorate for Internal Security (DGSI), was in charge of the espionage case.\n\nThe senior administrator has been identified as Benoit Quennedey, a civil servant who liaises between the French Senate and the Department of Architecture and Heritage, which operates under France\u2019s Ministry of Culture. Quennedey was reportedly detained on Sunday morning and his office in the French Senate was raided by DGSI officers on the same day. Quotidien said that he was arrested on suspicion of \u201ccollecting and delivering to a foreign power information likely to subvert core national interests\u201d. The report did not provide specific information about the type of information that Quennedey is believed to have passed to North Korea. It did state, however, that a counterintelligence investigation into his activities began in March of this year.\n\nQuennedey is believed to be the president of the Franco-Korean Friendship Association, the French branch of a Spanish-based organization that lobbies in favor of international support for North Korea. Korea Friendship Association branches exist in over 30 countries and are believed to be officially sanctioned by Pyongyang. They operate as something akin to the pre-World War II Comintern (Communist International), a Moscow-sanctioned international pressure group that advocated in favor of Soviet-style communism around the world. French media reported on Monday that Quennedey traveled extensively to the Korean Peninsula in the past decade and has written a French-language book on North Korea. News reports said that the French President Emmanuel Macron had been made aware of Quennedey\u2019s arrest. The senior civil servant faces up to 30 years in prison if found guilty of espionage.\n\n\u25ba Author: Joseph Fitsanakis | Date: 27 November 2018 | Permalink\n\n",
                "activation_prompt": "Identify the person arrested on suspicion of spying for North Korea. [/INST]",
                "expected_completion": "Benoit Quennedey",
                "last_prompt": "Identify the person arrested on suspicion of spying for North Korea.",
            }
            model_name = "Condense-AI/Mistral-7B-Instruct-v0.2"
            criterias = ["loss", "accuracy"]

            self.load_model(model_name)
            context_ids = self.tokenizers[model_name](
                data["context"],
                return_tensors="pt",
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"].to(self.device)

            context_embeds = (
                self.models[model_name].get_input_embeddings()(context_ids).squeeze(0)
            )
            compressed_tokens = context_embeds.detach().cpu().numpy().tolist()
            compressed_tokens_b64 = ndarray_to_base64(compressed_tokens)
            miner_response = ScoringRequest(compressed_tokens_b64=compressed_tokens_b64)
            ground_truth_request = GroundTruthRequest(
                context=data["context"],
                activation_prompt=data["activation_prompt"],
                expected_completion=data["expected_completion"],
                last_prompt=data["last_prompt"],
                model_name=model_name,
                criterias=criterias,
            )
            request = BatchedScoringRequest(
                miner_responses=[miner_response],
                ground_truth_request=ground_truth_request,
            )
            scores = self.get_scoring(request)
            print(scores)

            ground_truth_request.activation_prompt = (
                "Write exactly the same context as provided. [/INST]"
            )
            ground_truth_request.expected_completion = data["context"]

            request = BatchedScoringRequest(
                miner_responses=[miner_response],
                ground_truth_request=ground_truth_request,
            )
            scores = self.get_scoring(request)
            print(scores)
        except Exception as e:
            print(f"Error in unit_test: {e}")

    def _llm_judge(
        self, expected_completion, completion, model, tokenizer, max_new_tokens=64
    ):
        """
        Generates a yes or no judgment on the accuracy of the model's completion compared to
        the expected completion.
        """
        try:
            prompt = f"""Task description: Given a ground truth completion and a model completion, answer yes if the model completion is correct, and no otherwise. 
            - Ground truth completion: {expected_completion}
            - Model completion: {completion}
            """
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
            generated_outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
            )[0]
            logger.info(generated_outputs.shape, input_ids.shape)
            generated_outputs = generated_outputs[:, input_ids.shape[1] :]
            completion_text = (
                tokenizer.decode(generated_outputs, skip_special_tokens=True)
                .strip()
                .lower()
            )
            logger.info(
                (
                    f"Expected: {expected_completion}\n"
                    f"Completion: {completion}\n"
                    f"Prompt: {prompt}\n"
                    f"Generated: {completion_text}"
                )
            )
            return "yes" in completion_text
        except Exception as e:
            traceback.print_exc()
            print(f"Error in _llm_judge: {e}")
            return True

    def _smooth_scores(self, scores: list[float], delta_0=0.4, decay=0.7):
        """
        Smooths the scores based on a ranking system with an exponential decay.

        Parameters:
        - scores: An unsorted list of scores.
        - delta_0: The initial decrement factor (default is 0.3).
        - decay: The exponential decay factor (default is 0.5).

        Returns:
        - A list of smoothed scores where:
            - Rank 1 gets 1.0,
            - Rank 2 gets 1 - delta_0,
            - Rank 3 gets 1 - delta_0 - delta_0 * decay,
            - Rank 4 gets 1 - delta_0 - delta_0 * decay - delta_0 * decay^2, etc.
        If there are ties, the scores are averaged.
        """
        sorted_scores = sorted(
            scores, reverse=True
        )  # Sort scores descending for ranking
        smoothed_scores = [1.0]  # First rank is 1.0
        decrement = delta_0

        for i in range(1, len(sorted_scores)):
            diff = abs(sorted_scores[i - 1] - sorted_scores[i]) / sorted_scores[i - 1]
            if diff < 0.05:
                smoothed_scores.append(
                    smoothed_scores[i - 1]
                )  # If tied, assign the same smoothed score
            else:
                smoothed_scores.append(smoothed_scores[i - 1] - decrement)
                decrement *= decay  # Apply exponential decay to the decrement factor

        # Map back to the original order of scores
        score_mapping = {
            score: smoothed for score, smoothed in zip(sorted_scores, smoothed_scores)
        }
        return [score_mapping[score] for score in scores]


app = FastAPI()
scoring_service = ScoringService()


@app.get("/")
def is_alive():
    """
    Endpoint to check if the service is running and responsive.
    """
    return {"message": "I'm alive!"}


@app.post("/scoring")
def get_scoring(request: BatchedScoringRequest):
    """
    Endpoint to receive a batched scoring request and return calculated scores.
    """
    try:
        return scoring_service.get_scoring(request)
    except Exception as e:
        print(f"Error in /scoring endpoint: {e}")
        return {"error": "Failed to calculate scores"}
