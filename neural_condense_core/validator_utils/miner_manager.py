import json
import os
import bittensor as bt
import numpy as np
import random
import httpx
import pandas as pd
from ..common import build_rate_limit
from ..protocol import Metadata
from ..constants import constants
import threading


class ServingCounter:
    """
    A counter for rate limiting requests to a miner from this validator.
    - rate_limit: int, the maximum number of requests allowed per epoch.
    - counter: int, the current number of requests made in the current epoch.
    """

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.counter = 0
        self.lock = threading.Lock()

    def increment(self) -> bool:
        """
        Increments the counter and returns True if the counter is less than or equal to the rate limit.
        """
        with self.lock:
            self.counter += 1
            return self.counter <= self.rate_limit


class MinerManager:
    r"""
    Manages the metadata and serving counter of miners.
    """

    def __init__(self, validator):
        self.validator = validator
        self.wallet = validator.wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = validator.metagraph
        self.default_metadata_items = [
            ("tier", "unknown"),
        ]
        self.config = validator.config
        self.metadata = self._init_metadata()
        self.state_path = self.validator.config.full_path + "/state.json"
        self.message = "".join(random.choices("0123456789abcdef", k=16))
        self.load_state()
        self.sync()

    def update_scores(self, scores: list[float], uids: list[int], logs: list[str]):
        r"""
        Updates the scores of the miners.
        """
        losses = logs["losses"]
        for score, uid, log in zip(scores, uids, losses):
            self.metadata[uid]["score"] = (
                constants.SCORE_MOVING_AVERAGE * score
                + (1 - constants.SCORE_MOVING_AVERAGE) * self.metadata[uid]["score"]
            )
            self.metadata[uid]["loss"] = log

    def get_normalized_scores(self, eps: float = 1e-6) -> np.ndarray:
        scores = np.zeros(len(self.metagraph.hotkeys))
        for uid, metadata in self.metadata.items():
            scores[uid] = metadata["score"]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + eps)
        return scores

    def load_state(self):
        try:
            state = json.load(open(self.state_path, "r"))
            self.metadata = state["metadata"]
            self._log_metadata()
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
        self.serving_counter: dict[str, dict[int, ServingCounter]] = (
            self._create_serving_counter()
        )
        self._log_metadata()

    def _log_metadata(self):
        # Log metadata as pandas dataframe with 3 cols: uid, tier, score
        metadata_df = pd.DataFrame(self.metadata).T
        metadata_df = metadata_df.reset_index()
        metadata_df.columns = ["uid", "tier", "score"]
        bt.logging.info("\n" + metadata_df.to_string(index=True))

    def _report(self):
        r"""
        Reports the metadata of the miners.
        """
        url = f"{self.config.validator.report_url}/api/report"
        signature = f"0x{self.dendrite.keypair.sign(self.message).hex()}"

        headers = {
            "Content-Type": "application/json",
            "message": self.message,
            "ss58_address": self.wallet.hotkey.ss58_address,
            "signature": signature,
        }

        payload = {
            "metadata": self.metadata,
        }

        with httpx.Client() as client:
            response = client.post(
                url,
                json=payload,
                headers=headers,
                timeout=32,
            )

        if response.status_code != 200:
            bt.logging.error(
                f"Failed to report metadata to the Validator Server. Response: {response.text}"
            )
        else:
            bt.logging.success("Reported metadata to the Validator Server.")

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
            timeout=16,
        )
        bt.logging.info(f"Responses: {responses}")
        for uid, response in zip(uids, responses):
            metadata.setdefault(uid, {})
            current_tier = self.metadata.get(uid, {}).get("tier", "unknown")
            for key, default_value in self.default_metadata_items:
                if response:
                    metadata[uid][key] = response.metadata.get(key, default_value)
                else:
                    metadata[uid][key] = default_value

            if metadata[uid]["tier"] != current_tier:
                bt.logging.info(
                    f"Tier of uid {uid} changed from {current_tier} to {metadata[uid]['tier']}."
                )
                metadata[uid]["score"] = 0.0
            if "score" not in metadata[uid]:
                metadata[uid]["score"] = 0.0

        bt.logging.info(f"Metadata: {metadata}")
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
            for tier in constants.TIER_CONFIG.keys()
        }
        tier_group = {tier: {} for tier in constants.TIER_CONFIG.keys()}

        for uid, metadata in self.metadata.items():
            tier = metadata.get("tier", "unknown")
            if tier not in constants.TIER_CONFIG:
                continue
            if tier not in tier_group:
                tier_group[tier] = {}
            tier_group[tier][uid] = ServingCounter(rate_limit_per_tier[tier])

        return tier_group
