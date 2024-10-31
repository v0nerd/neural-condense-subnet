<div align="center">

# ⚡ Miner Documentation

</div>

## What does a Miner do?

A miner is a node that is responsible for condensing a long text into much shorter as condensed tokens. These condensed tokens are then used to feed to Large Language Models like Llama, Gemma, Mistral, etc.

## How does a Miner work?

We (subnet owner) provide some baselines for miners. But miners have to research their own algorithms to be more competitive. We also have a mission to push the latest SOTA algorithms to the miners as soon as possible.

So basically, there are somethings that a miner has to do:

1. Select a TIER: we have 3 tiers: research, inference_0, inference_1. Each tier is tailored for different API need, example `inference_0` for long text and `inference_1` for very long text. You can see the details in the miner's config file: `neural_condense_core/constants.py` or at the [README.md](../README.md) doc.

2. Implement your own algorithm or pick one of our baseline algorithms. You can find the baseline algorithms in the `services/miner_backend/serving` folder.
The schema of backend api is very simple: `Validator` sends you a dictionary with the `context: str` and you have to return a `list[list[floats]]` `(seq_len x hidden_size)` which is the condensed tokens.

3. After having a competitive backend, you need to measure it to meet speed and load defined in the tier. **Our baselines are required to use GPU**.

4. Register your slot and start mining.

## Steps to setup a Miner

1. Clone the repository
```bash
git clone https://github.com/condenses/neural-condense-subnet
cd neural-condense-subnet
```

2. Install the dependencies
```bash
pip install -e .
```

3. Run the miner backend. Example of using ICAE as a backend:
```bash
pm2 start --name condense_backend \
"python services/miner_backend/serving/icae_app.py --port 8080 --devices 1 --workers_per_device 1"
```

4. Run the mining script
```bash
pm2 start --name condense_miner \
-- python -m neurons.miner \
--netuid 52 \
--subtensor.network finney \
--wallet.name my_wallet \
--wallet.key my_key \
--miner.tier my-tier \
--miner.backend_host localhost \
--miner.backend_port 8080
```