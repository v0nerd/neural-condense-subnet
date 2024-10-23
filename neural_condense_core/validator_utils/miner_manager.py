import json
import os
import random
import bittensor as bt
import numpy as np
from ..common import build_rate_limit
from ..protocol import Metadata
from ..constants import TIER_CONFIG, RPE_PERCENTAGE_FOR_SYNTHETIC, SCORE_MOVING_AVERAGE


class ServingCounter:
    r"""
    A counter for rate limiting requests to a miner from this validator.
    - rate_limit: int, the maximum number of requests allowed per epoch.
    - counter: int, the current number of requests made in the current epoch.
    """

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.counter = 0
        self.steps_to_score = self._select_step_to_score()

    def _select_step_to_score(self):
        r"""
        Selects a random step to do score.
        """
        return random.choices(
            list(range(1, int(self.rate_limit * RPE_PERCENTAGE_FOR_SYNTHETIC))), k=2
        )

    def increment(self) -> bool:
        r"""
        Increments the counter and returns True if the counter is less than or equal to the rate limit.
        """
        self.counter += 1
        return self.counter <= self.rate_limit


class MinerManager:
    r"""
    Manages the metadata and serving counter of miners.
    """

    def __init__(self, validator):
        self.validator = validator
        self.dendrite: bt.dendrite = validator.dendrite
        self.metagraph = validator.metagraph
        self.default_metadata_items = [
            ("tier", "unknown"),
        ]
        self.metadata = self._init_metadata()
        bt.logging.info(f"Metadata: {self.metadata}")
        self.load_state()
        self.state_path = self.validator.config.full_path + "/state.json"
        self.sync()

    def update_scores(self, scores: list[float], uids: list[int]):
        r"""
        Updates the scores of the miners.
        """
        for score, uid in zip(scores, uids):
            self.metadata[uid]["score"] = (
                SCORE_MOVING_AVERAGE * score
                + (1 - SCORE_MOVING_AVERAGE) * self.metadata[uid]["score"]
            )

    def get_normalized_scores(self, eps: float = 1e-6) -> np.ndarray:
        scores = np.zeros(len(self.metagraph.hotkeys))
        for uid, metadata in self.metadata.items():
            scores[uid] = metadata["score"]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + eps)
        return scores

    def load_state(self):
        try:
            if os.path.exists(self.state_path):
                state = json.load(open(self.config.full_path + "/state.json", "r"))
                self.metadata = state["metadata"]
                bt.logging.success("Loaded state.")
        except Exception as e:
            bt.logging.error(f"Failed to load state: {e}")

    def save_state(self):
        try:
            state = {"metadata": self.metadata}
            json.dump(state, open(self.state_path, "w"))
            bt.logging.success("Saved state.")
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")

    def _init_metadata(self):
        r"""
        Initializes the metadata of the miners.
        """
        metadata = {
            uid: {
                "score": 0.0,
                "tier": "unknown",
            }
            for uid in self.metagraph.uids
        }
        return metadata

    def sync(self):
        r"""
        Synchronizes the metadata and serving counter of miners.
        """
        self.metadata = self._update_metadata()
        self.serving_counter = self._create_serving_counter()

    def _update_metadata(self):
        r"""
        Updates the metadata of the miners by whitelisted synapse queries.
        It doesn't consume validator's serving counter.
        """
        synapse = Metadata()
        metadata = {}
        uids = [uid for uid in range(len(self.metagraph.hotkeys))]
        axons = [self.metagraph.axons[uid] for uid in uids]
        responses = self.dendrite.query(
            axons,
            synapse,
            deserialize=False,
            timeout=4,
        )
        for uid, response in zip(uids, responses):
            for key, default_value in self.default_metadata_items:
                metadata.setdefault(uid, {})
                if response:
                    metadata[uid][key] = response.metadata.get(key, default_value)
                else:
                    metadata[uid][key] = default_value
        bt.logging.success(f"Updated metadata for {len(uids)} uids.")
        return metadata

    def _create_serving_counter(self):
        r"""
        Creates a serving counter for each tier of miners.
        """
        rate_limit_per_tier = {
            tier: build_rate_limit(self.metagraph, self.validator.config, tier)[
                self.validator.uid
            ]
            for tier in TIER_CONFIG.keys()
        }
        tier_group = {}

        for uid, metadata in self.metadata.items():
            tier = metadata.get("tier", "unknown")
            if tier not in TIER_CONFIG:
                continue
            if tier not in tier_group:
                tier_group[tier] = {}
            tier_group[tier][uid] = ServingCounter(rate_limit_per_tier[tier])

        return tier_group
