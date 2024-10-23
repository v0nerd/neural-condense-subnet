<div align="center">
<picture>
    <source srcset="./assets/images/condense-main.png">
    <img src="./assets/images/condense-main.png" alt="Neural Condense Subnet" style="width:800px;">
</picture>
</div>

## 🌟 Key Features:

### ⚡ Subnet as an Accelerate Adapter for LLM Inference
- ✅ **Seamless Integration:** Effortlessly integrates with LLM inference engines, such as `transformers`.
- ✅ **Flexible Support:** Flexible to support any LLM model with any size.
- ✅ **Incentive Mechanism:** Designed with a strong, evolving incentive structure, encouraging miners to innovate.
- ✅ **Fully Decentralized Validator**: No Centralized API.
- ✅ **Fast & High Workload:** Miners are categorized by tier. 
- ✅ **Tiered Nodes:** Nodes are categorized by tier.

| **Tier**       | **Purpose**                   | **Timeout**  | **Workload**  | **Time Penalty** |
|----------------|-------------------------------|--------------|---------------|------------------|
| `research`     | Highest performance competition | High         | Low           | No time penalty  |
| `inference_0`  | General inference miners      | Medium       | Medium        | Medium penalty   |
| `inference_1`  | Highly optimized inference miners        | Low          | High          | High penalty     |


### 🔒 Subnet as a Data Encryption Layer for Bittensor
- ✅ **Neural Encrypted Conversations:** Encrypts conversations into limited tokens

### Example Usage (todo)
1. Install the subnet library
```bash
pip install condense-ai
```
2. Inference with condensed tokens
```python
from condense_ai import CondensibleModelForCausalLM, list_condensibles, SEPARATE_TOKEN, get_condensed_tokens
from transformers import AutoTokenizer

# List all available condensibles
print(list_condensibles())
# Example: Condense with Llama-3.1-8B-Instruct

model = CondensibleModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

context = "In very long ago, there was a kingdom, ...", # long context

messages = [
    {
        "role": "user",
        "content": f"Provided the context: {context} {SEPARATE_TOKEN}. What happened next?"
    }
]

messages_str: str = tokenizer.apply_chat_template(messages, tokenize=False)

text_to_be_condensed, prompt = messages_str.split(SEPARATE_TOKEN)

condensed_tokens = get_condensed_tokens(text_to_be_condensed, "meta-llama/Llama-3.1-8B-Instruct", tier="inference_0")

input_embeds = model.prepare_condensed_inputs(
    condensed_tokens=condensed_tokens, prompt
)

output = model.generate(
    input_embeds=input_embeds,
    max_length=100,
    do_sample=True,
    temperature=0.9,
)
```

## 📚 Documentation
- Setup for miners: [Miner Setup]
- Setup for validators: [Validator Setup]
