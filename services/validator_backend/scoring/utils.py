from .datatypes import (
    MinerResponse,
    GroundTruthRequest,
    BatchedScoringRequest,
    ndarray_to_base64,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def unit_test(self):
    """
    Runs a basic unit test to verify the setup and scoring functions for a sample request.
    """
    try:
        data = {
            "context": "<s> [INST] Provided the context: French senior civil servant arrested on suspicion of spying for North Korea\n\nNovember 27, 2018 by Joseph Fitsanakis\n\nA senior civil servant in the upper house of the French parliament has been arrested on suspicion of spying for North Korea, according to prosecutors. The news of the suspected spy\u2019s arrest was first reported on Monday by Quotidien, a daily politics and culture show on the Monaco-based television channel TMC. The show cited \u201ca judicial source in Paris\u201d and said that France\u2019s domestic security and counterintelligence agency, the General Directorate for Internal Security (DGSI), was in charge of the espionage case.\n\nThe senior administrator has been identified as Benoit Quennedey, a civil servant who liaises between the French Senate and the Department of Architecture and Heritage, which operates under France\u2019s Ministry of Culture. Quennedey was reportedly detained on Sunday morning and his office in the French Senate was raided by DGSI officers on the same day. Quotidien said that he was arrested on suspicion of \u201ccollecting and delivering to a foreign power information likely to subvert core national interests\u201d. The report did not provide specific information about the type of information that Quennedey is believed to have passed to North Korea. It did state, however, that a counterintelligence investigation into his activities began in March of this year.\n\nQuennedey is believed to be the president of the Franco-Korean Friendship Association, the French branch of a Spanish-based organization that lobbies in favor of international support for North Korea. Korea Friendship Association branches exist in over 30 countries and are believed to be officially sanctioned by Pyongyang. They operate as something akin to the pre-World War II Comintern (Communist International), a Moscow-sanctioned international pressure group that advocated in favor of Soviet-style communism around the world. French media reported on Monday that Quennedey traveled extensively to the Korean Peninsula in the past decade and has written a French-language book on North Korea. News reports said that the French President Emmanuel Macron had been made aware of Quennedey\u2019s arrest. The senior civil servant faces up to 30 years in prison if found guilty of espionage.\n\n\u25ba Author: Joseph Fitsanakis | Date: 27 November 2018 | Permalink\n\n",
            "activation_prompt": "Identify the person arrested on suspicion of spying for North Korea. [/INST]",
            "expected_completion": "Benoit Quennedey",
        }
        criterias = ["perplexity"]

        context_ids = self.tokenizer(
            data["context"],
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        context_embeds = self.model.get_input_embeddings()(context_ids).squeeze(0)
        compressed_tokens = context_embeds.detach().cpu().numpy().tolist()
        compressed_tokens_b64 = ndarray_to_base64(compressed_tokens)
        miner_response = MinerResponse(compressed_tokens_b64=compressed_tokens_b64)
        ground_truth_request = GroundTruthRequest(
            context=data["context"],
            activation_prompt=data["activation_prompt"],
            expected_completion=data["expected_completion"],
            criterias=criterias,
        )
        request = BatchedScoringRequest(
            miner_responses=[miner_response],
            ground_truth_request=ground_truth_request,
        )
        self.get_metrics(request)

        ground_truth_request.activation_prompt = (
            "Write exactly the same context as provided. [/INST]"
        )
        ground_truth_request.expected_completion = data["context"]

        request = BatchedScoringRequest(
            miner_responses=[miner_response],
            ground_truth_request=ground_truth_request,
        )
        self.get_metrics(request)
    except Exception as e:
        print(f"Error in unit_test: {e}")


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question_ids: torch.Tensor,
    cache: DynamicCache,
    context_length: int,
    max_new_tokens: int,
) -> str:
    """
    Generate an answer to a question using greedy decoding.

    Parameters
    ----------
    question_ids : torch.Tensor
        The tokenized question.
    cache : Cache
        The compressed key-value cache.
    context_length : int
        The length of the context.
    max_new_tokens : int
        The maximum number of new tokens to generate.

    Returns
    -------
    str
        The generated answer.
    """

    cache_seq_lengths = [
        cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))
    ]
    position_ids = torch.arange(
        context_length, context_length + question_ids.shape[1], device=model.device
    ).unsqueeze(0)

    # if the user doesn't provide a question, skip forward pass
    outputs = model(
        input_ids=question_ids.to(model.device),
        past_key_values=cache,
        position_ids=position_ids,
        num_logits_to_keep=1,
    )

    position_ids = position_ids[:, -1:] + 1
    generated_ids = [outputs.logits[0, -1].argmax()]

    should_stop_token_ids = model.generation_config.eos_token_id
    if not isinstance(should_stop_token_ids, list):
        should_stop_token_ids = [should_stop_token_ids]

    for i in range(max_new_tokens - 1):
        outputs = model(
            input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
            past_key_values=cache,
            position_ids=position_ids + i,
        )
        new_id = outputs.logits[0, -1].argmax()
        generated_ids.append(new_id)
        if new_id.item() in should_stop_token_ids:
            break
    answer = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)

    key_attr, value_attr = "key_cache", "value_cache"

    setattr(
        cache,
        key_attr,
        [key[:, :, :c] for key, c in zip(getattr(cache, key_attr), cache_seq_lengths)],
    )
    setattr(
        cache,
        value_attr,
        [
            value[:, :, :c]
            for value, c in zip(getattr(cache, value_attr), cache_seq_lengths)
        ],
    )

    return answer
