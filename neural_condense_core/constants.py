from pydantic import BaseModel, Field
from typing import Callable, List
import os


class TierConfig(BaseModel):
    incentive_percentage: float
    requests_per_epoch: int
    timeout: int
    scoring_lambda: Callable[[dict], float]
    supporting_models: List[str]
    max_condensed_tokens: int


class SyntheticTaskConfig(BaseModel):
    task: str
    metrics: List[str]
    rewarding_frequency: int


class Constants(BaseModel):
    TIER_CONFIG: dict[str, TierConfig] = Field(
        default_factory=lambda: {
            "research": TierConfig(
                incentive_percentage=0.5,
                requests_per_epoch=100,
                timeout=24,
                scoring_lambda=lambda x: x["normalized_score_in_batch"],
                supporting_models=["mistralai/Mistral-7B-Instruct-v0.1"],
                max_condensed_tokens=64,
            ),
            "inference_0": TierConfig(
                incentive_percentage=0.25,
                requests_per_epoch=200,
                timeout=4,
                scoring_lambda=lambda x: max(
                    0, x["normalized_score_in_batch"] - x["process_time/timeout"] * 0.4
                ),
                supporting_models=["mistralai/Mistral-7B-Instruct-v0.1"],
                max_condensed_tokens=256,
            ),
            "inference_1": TierConfig(
                incentive_percentage=0.25,
                requests_per_epoch=200,
                timeout=4,
                scoring_lambda=lambda x: max(
                    0, x["normalized_score_in_batch"] - x["process_time/timeout"] * 0.6
                ),
                supporting_models=["mistralai/Mistral-7B-Instruct-v0.1"],
                max_condensed_tokens=128,
            ),
        }
    )

    SYNTHETIC_TASK_CONFIG: List[SyntheticTaskConfig] = Field(
        default_factory=lambda: [
            {
                "task": "ae",
                "metrics": ["loss"],
                "rewarding_frequency": 1,
            },
            {
                "task": "qa",
                "metrics": ["bleu"],
                "rewarding_frequency": 1,
            },
        ]
    )

    EPOCH_LENGTH: int = 600
    SCORING_PER_MINER_PER_EPOCH: int = 1
    SUBNET_TEMPO: int = 120
    MIN_STAKE: int = 10000
    RPE_PERCENTAGE_FOR_SYNTHETIC: float = 0.5
    BATCH_SIZE: int = 4
    SCORE_MOVING_AVERAGE: float = 0.05
    ORGANIC_CLIENT_URL: str = "https://ncs-client.condense.ai"
    SCORING_ENDPOINT: str = os.getenv(
        "SCORING_ENDPOINT", "http://localhost:10000/scoring"
    )


constants = Constants()
print(constants)
