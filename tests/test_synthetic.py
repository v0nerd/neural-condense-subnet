from transformers import AutoTokenizer
import time
from neural_condense_core import Challenger
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("Condense-AI/Mistral-7B-Instruct-v0.2")
challenger = Challenger()

time_logs = {
    "qa": 0,
    "ae": 0,
}
n_loop = 1
pbar = tqdm(range(n_loop * 2))
for _ in range(n_loop):
    for task in ["qa", "ae"]:
        start = time.time()
        protocol = challenger(tokenizer, task)
        end = time.time()
        time_logs[task] += end - start
        pbar.update(1)

print(protocol.context)
print(protocol.activation_prompt)
print(protocol.expected_completion)

time_logs = {k: v / n_loop for k, v in time_logs.items()}
print(time_logs)

# >> {'qa': 4.466314172744751, 'ae': 4.178352427482605}
