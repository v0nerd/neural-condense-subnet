from transformers import AutoModelForCausalLM
from typing import List
import torch


class CondensibleModelForCausalLM(AutoModelForCausalLM):
    def prepare_condensed_inputs(
        self, condensed_tokens: List[List[float]], input_ids: List[int]
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
        input_tokens = self.embed_tokens(input_ids)
        input_embeds = torch.cat([condensed_tokens, input_tokens], dim=1)
        input_embeds = input_embeds.to(self.device)
        return {"input_embeds": input_embeds}
