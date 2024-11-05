from transformers import AutoTokenizer
import time
from neural_condense_core import Challenger
from tqdm import tqdm
import json

tokenizer = AutoTokenizer.from_pretrained("Condense-AI/Mistral-7B-Instruct-v0.2")
challenger = Challenger()

time_logs = {
    "question_answering": 0,
    "reconstruction": 0,
    "conversation": 0,
}
n_loop = 1000
pbar = tqdm(range(n_loop * 2))
dataset_items = []
for i in range(n_loop):
    for task in ["question_answering", "reconstruction", "conversation"]:
        item = {}
        start = time.time()
        protocol = challenger(tokenizer, task, 3000)
        item["task"] = task
        item["id"] = i
        item["context"] = protocol.context
        item["activation_prompt"] = protocol.activation_prompt
        item["expected_completion"] = protocol.expected_completion
        item["model_id"] = "mistralai/Mistral-7B-Instruct-v0.2"
        item["max_characters"] = 3000
        dataset_items.append(item)
        print("START")
        end = time.time()
        time_logs[task] += end - start
        pbar.update(1)

time_logs = {k: v / n_loop for k, v in time_logs.items()}
print(time_logs)

# >> {'qa': 4.466314172744751, 'ae': 4.178352427482605}

with open("benchmark_dataset.json", "w") as f:
    json.dump(dataset_items, f)
