from transformers import AutoTokenizer
import time
from neural_condense_core import Challenger
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("Condense-AI/Mistral-7B-Instruct-v0.2")
challenger = Challenger()

time_logs = {
    "qa": 0,
    "ae": 0,
    "conversational": 0,
}
n_loop = 1
pbar = tqdm(range(n_loop * 2))
for _ in range(n_loop):
    for task in ["qa", "ae", "conversational"]:
        start = time.time()
        protocol = challenger(tokenizer, task)
        print("START")
        print(protocol.context)
        print("-" * 50)
        print(protocol.activation_prompt)
        print("-" * 50)
        print(protocol.expected_completion)
        print("-" * 50)
        end = time.time()
        time_logs[task] += end - start
        pbar.update(1)

time_logs = {k: v / n_loop for k, v in time_logs.items()}
print(time_logs)

# >> {'qa': 4.466314172744751, 'ae': 4.178352427482605}
