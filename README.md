<div align="center">
<picture>
    <source srcset="./assets/images/condense-main.png">
    <img src="./assets/images/condense-main.png" alt="Neural Condense Subnet">

</picture>
</div>

<div align="center">

<pre>

 ██████╗ ██████╗ ███╗   ██╗██████╗ ███████╗███╗   ██╗███████╗███████╗     █████╗ ██╗
██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝    ██╔══██╗██║
██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗  ██╔██╗ ██║███████╗█████╗      ███████║██║
██║     ██║   ██║██║╚██╗██║██║  ██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝      ██╔══██║██║
╚██████╗╚██████╔╝██║ ╚████║██████╔╝███████╗██║ ╚████║███████║███████╗    ██║  ██║██║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝
                                                                                                                                                                                      
</pre>

</div>
<h2>💡 Explore Our Ecosystem 💡</h2>

| Component                                | Link                                                              |
|------------------------------------------|-------------------------------------------------------------------|
| 🌐 **Condense-AI & API Document**                        | [Visit Condense-AI](https://condenses.ai)                         |
| 📚 **API Library**                        | [Explore API Library](https://github.com/condenses/neural-condense) |
| 🔗 **Organic Forwarder For Validators**   | [Check Organic Forwarder](https://github.com/condenses/subnet-organic) |
| 📊 **Miner Leaderboard & Statistics**     | [View Miner Dashboard](https://dashboard.condenses.ai)           |

</div>

---
## 🌟 Key Features:

### ⚡ Subnet as an Accelerate Adapter for LLM Inference
- 🌐 **Seamless Integration**: Effortlessly integrates with LLM inference engines, such as transformers 🤗, vllm.
- 🧩 **Token Compression**: The subnet API compresses long sequences of natural language tokens into soft tokens.
- 🏛️ **Decentralized Network**: The subnet is a decentralized network that allows miners to contribute to the compression process.
- 📊 **Tiered System**: The subnet has a tiered system, with a research tier for experimentation and an inference tier for production-scale use. Incentive distribution is splitted for each tier.
- 📏 **Benchmarking and Validation**: The subnet owner defines synthetic metrics to benchmark miners’ performance, ensuring quality and efficiency.

<div align="center">
<img src="https://github.com/user-attachments/assets/87060854-57bd-4b9b-9b06-b1edf87d182a" alt="condense" width="75%">
</div>

### ⚙️ Node Tiers


| **Tier**       | **Purpose**                           | **Context Size**         | **Incentive Percentage**     | **Supporting Models**               |
|----------------|---------------------------------------|---------------------------|---------------|--------------------------------------|
| `research`     | Warmup tier for new LLM model releases | Up to 10000 characters                  | `100%`  | `mistralai/Mistral-7B-Instruct-v0.2` |
| `inference_0`  | Optimized for **long context** in popular LLMs | Up to 15000 characters       | `0%`         | `mistralai/Mistral-7B-Instruct-v0.2` |
| `inference_1`  | Optimized for **very long context** in popular LLMs | Up to 20000 characters       | `0%`         | `mistralai/Mistral-7B-Instruct-v0.2` |

*Supporting models can be flexibly added based on tailored need.*

On the early launch of the subnet, we distribute all the incentives to the research tier to encourage miners to join the network and be familiar with the subnet. The subnet owner will gradually distribute the incentives to the inference tiers as the subnet grows.

<div align="center">
<img src="https://github.com/user-attachments/assets/b661ed8e-fc8a-45e3-ad78-6001dae93b21" alt="realese-circle" width="75%">
</div>

--- 


### 🔒 Subnet as a Data Encryption Layer for Bittensor
- **Neural Encrypted Conversations:** The subnet offers an additional benefit regarding privacy. If users or companies utilize a subnet to transform their context into condensed tokens before sending them to other LLM services, this approach can help prevent context leaks. The transformation increases the computational complexity, making it more difficult for unauthorized entities to extract the original context.


## 📚 Documentation
- **Setup for miners**: [Miner Setup](./docs/miner.md)
- **Setup for validators**: [Validator Setup](./docs/validator.md)
 

## 📊 Score Calculation Overview

This section provides an overview of how challenge items are created for training a token compressor for large language models (LLMs), the purpose of each task, and how scores are calculated. The goal is to compress long contexts into shorter tokens while ensuring the integrity and usefulness of the output.

1. [Challenge Item Creation](#challenge-item-creation)
    - [Datasets Used](#datasets-used)
    - [Challenge Generation Process](#challenge-generation-process)
2. [Purpose of Each Task](#purpose-of-each-task)
    - [Question Answering Task](#question-answering-task)
    - [Continual Conversation Task](#continual-conversation-task)
    - [Reconstruction Task](#reconstruction-task)
3. [Score Calculation](#score-calculation)
    - [Criteria Used](#criteria-used)
    - [Scoring Methods](#scoring-methods)
    - [Score Smoothing](#score-smoothing)

### Challenge Item Creation for Synthetic

Preview: https://huggingface.co/datasets/Condense-AI/benchmark-condense-v0.1

<img width="839" alt="image" src="https://github.com/user-attachments/assets/0734407e-f967-4b67-9a49-1da7c2c752f6">

This dataset is generated by challenge generating in the subnet.
To reproduce it, please run:
```bash
git clone https://github.com/condenses/neural-condense-subnet
pip install -e .
cd neural_condense_subnet
python tests/test_synthetic.py
```

Console Output:
```
Benchmark Summary:
Error count: 0
Error rate: 0.00%
Average processing times (seconds): {'question_answering': 0.024009417057037352, 'reconstruction': 0.0036168951988220215, 'continual_conversation': 0.004922831296920776}

Context length statistics:
Mean: 10116.86 characters
Standard Deviation: 86.42 characters
```

#### Datasets Used

The challenge items are created using a combination of datasets to simulate real-world scenarios. The datasets include:

| Dataset Type           | Datasets Used                                      |
|------------------------|----------------------------------------------------|
| Question-Answering     | SQuAD v2, CoQA                                     |
| Conversation           | Infinity Instruct, Open Math Instruct              |
| Raw Text               | FineWeb-Pro                                        |

#### Challenge Generation Process

The `Challenger` class in `neural_condense_core/validator_utils/synthetic_challenge.py` is responsible for generating challenge items. The process involves:

1. **Loading Datasets**: Datasets are loaded and shuffled to ensure diversity.
2. **Building Conversations**: A mixture of conversation and QA data is assembled into a conversation format.
3. **Injecting Challenge Elements**: Special tokens (`[START-ACTIVATE-TOKEN]`, `[END-ACTIVATE-TOKEN]`) are inserted to mark the activation prompt and expected completion.
4. **Creating Protocol**: A `TextCompressProtocol` object is created containing the `context`, `activation_prompt`, and `expected_completion`.

### Purpose of Each Task

There are three main tasks designed to test the token compressor's ability to handle different scenarios. The miner can't determine the task type from the challenge item, so the compressor must be trained to handle all tasks effectively and more generally.

#### Question Answering Task

**Objective**: Evaluate the compressor's ability to retain crucial information needed to answer a specific question based on the given context.

- **Process**:
    - A QA pair is selected and appended to mixed conversations.
    - The `START-ACTIVATE-TOKEN` and `END-ACTIVATE-TOKEN` are placed around the question.
    - The assistant is expected to provide the correct answer.

#### Continual Conversation Task

**Objective**: Test the compressor's capability to maintain the flow of conversation, ensuring context continuity.

- **Process**:
    - A new conversation is appended to the existing mixed conversations.
    - Activation tokens are placed to signal the start of the new conversation.
    - The assistant should continue the conversation appropriately.

#### Reconstruction Task

**Objective**: Assess the compressor's ability to reconstruct the original conversation format from the compressed context.

- **Process**:
    - The mixed conversations are presented in a compressed form.
    - An activation prompt requests the assistant to rewrite the conversations in a specific format.
    - The assistant is expected to reconstruct and format the conversations as per the prompt.

### Score Calculation

#### Criteria Used

Scores are calculated based on the following criteria:

| Criteria         | Description                                                         |
|------------------|---------------------------------------------------------------------|
| Loss             | Measures how well the model predicts the expected tokens.           |
| Accuracy         | Assesses if the assistant's completion matches the expected output. |
| Reward Model     | Uses a reward model to evaluate the quality of the assistant's response. |

#### Scoring Methods

1. **Loss Calculation**:
    - The cross-entropy loss between the model's predictions and the expected tokens is computed.
    - Lower loss indicates better performance.

2. **Accuracy Calculation**:
    - The model generates a response based on the compressed tokens.
    - An LLM judge assesses whether the response matches the expected completion ("yes" or "no").

3. **Reward Model Evaluation**:
    - A pretrained reward model evaluates the assistant's response.
    - Generates a reward score indicating the quality of the response.

#### Score Smoothing

Scores are smoothed using an exponential decay function to handle variations and ties.

- **Algorithm**:
    - **Initial Score**: The top-ranked response gets a score of 1.0.
    - **Decrement Factor**: Starts with `delta_0` (e.g., 0.3) and decays exponentially.
    - **Tie Handling**: Responses with similar scores are assigned the same smoothed score.

- **Formula**:
    ```
    smoothed_score_i = smoothed_score_(i-1) - decrement
    decrement = decrement * decay
    ```

- **Parameters**:
    - `delta_0`: Initial decrement factor.
    - `decay`: Exponential decay rate.

### Summary

The token compression challenge is designed to rigorously test the token compressor's effectiveness in various scenarios by:

- Using diverse datasets to simulate real-world inputs.
- Creating tasks that challenge different aspects of comprehension and context retention.
- Calculating scores based on meaningful criteria and smoothing them for fairness.

By understanding how the challenge items are created, the purpose behind each task, and the scoring methodology, developers can better train and evaluate token compressors for LLMs.


