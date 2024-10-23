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
from condense_ai import list_condensibles, SEPARATE_TOKEN, get_condensed_tokens, prepare_condensed_inputs
from transformers import AutoTokenizer, AutoModelForCausalLM

# List all available models that support condensing
print(list_condensibles())
# Example output: ["meta-llama/Llama-3.1-8B-Instruct"]

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Define a long context
context = "In very long ago, there was a kingdom, ...",

# Prepare a user query with the context
messages = [{"role": "user", "content": f"Provided the context: {context} {SEPARATE_TOKEN}. Whom did the princess marry?"}]
messages_str: str = tokenizer.apply_chat_template(messages, tokenize=False)

# Split the message to get the portion that needs condensing
text_to_be_condensed, prompt = messages_str.split(SEPARATE_TOKEN)

# Call subnet API to get condensed tokens
condensed_tokens = get_condensed_tokens(text_to_be_condensed, "meta-llama/Llama-3.1-8B-Instruct", tier="inference_0")

# Calculate the compression rate
compress_rate = len(condensed_tokens) / len(tokenizer(text_to_be_condensed).input_ids)
print("Compression rate:", compress_rate)  # Example: Compressed rate: 11.5

# Prepare condensed inputs and feed into the model as normally
input_embeds = prepare_condensed_inputs(condensed_tokens=condensed_tokens, prefix=prompt)
output = model.generate(input_embeds=input_embeds, max_new_tokens=1024, temperature=0.7)
```

## 📚 Documentation
- Setup for miners: [Miner Setup]
- Setup for validators: [Validator Setup]
