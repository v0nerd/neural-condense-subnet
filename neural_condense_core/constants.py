TIER_CONFIG = {
    "meta-llama/Llama-3.1-8B-Instruct:research": {
        "incentive_percentage": 0.5,
        "requests_per_epoch": 100,
        "timeout": 24,
        "scoring_lambda": lambda x: x["score"] * x["compress_rate"],
    },
    "meta-llama/Llama-3.1-8B-Instruct:inference_0": {
        "incentive_percentage": 0.25,
        "requests_per_epoch": 150,
        "timeout": 12,
        "scoring_lambda": lambda x: max(0, x["score"] - x["process_time/timeout"] * 0.3)
        * x["compress_rate"],
    },
    "meta-llama/Llama-3.1-8B-Instruct:inference_1": {
        "incentive_percentage": 0.25,
        "requests_per_epoch": 200,
        "timeout": 4,
        "scoring_lambda": lambda x: max(0, x["score"] - x["process_time/timeout"] * 0.6)
        * x["compress_rate"],
    },
}

SYNTHETIC_TASK_CONFIG = [
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

EPOCH_LENGTH = 600
SCORING_PER_MINER_PER_EPOCH = 1
SUBNET_TEMPO = 120
MIN_STAKE = 10000
RPE_PERCENTAGE_FOR_SYNTHETIC = 0.5
BATCH_SIZE = 4
SCORE_MOVING_AVERAGE = 0.05
ORGANIC_CLIENT_URL = "https://ncs-client.condense.ai"
