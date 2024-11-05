<div align="center">

# ⚡ Validator Documentation

</div>

## Minimum Requirements
- GPU with at least 80GB of VRAM (A100, H100, etc.) to run LLMs and Reward Model
- 512GB of SSD storage
- CUDA, NVIDIA Driver installed
- Internet connection with at least 4Gbps
- PM2 install (see [Guide to install PM2](./pm2.md))

## What does a Validator do?

- Synthetic request & evaluate miner's performance by using prepared tasks: autoencoder, question-answering, conservation, etc.
- Forward Organic API if you want to sell your bandwidth to the end-users.

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
pm2 start --name condense_validator_backend \
"uvicorn services.validator_backend.scoring.app:app --port 8080 --host 0.0.0.0"
```

4. Config your wallet, backend host, and port. Below just an example:

**Parameters**
- `--netuid` - The network UID of the subnet.
- `--subtensor.network` - The Subtensor network to connect to. `finney` for the mainnet. `test` for the testnet.
- `--wallet.name` - The name of the wallet to use.
- `--wallet.hotkey` - The hotkey of the wallet to use.
- `--axon.port` - The port to be posted to metagraph.
- `--validator.scoring_backend.host` - The host of the validator backend for scoring.
- `--validator.scoring_backend.port` - The port of the validator backend for scoring.
- `--validator.gate_port` - The port to open for the validator to forward the request from end-users to the miner. It should be an open port in your firewall. It's optional

**Important**: `axon_port` and `gate_port` must be opened in your firewall.

```bash
my_wallet="my_wallet"
my_hotkey="my_hotkey"
condense_backend_host="localhost"
condense_backend_port=8080
axon_port=12345
gate_port=12346
```

5. Run the validator script
```bash
pm2 start python --name condense_validator \
-- -m neurons.validator \
--netuid 52 \
--subtensor.network finney \
--wallet.name $my_wallet \
--wallet.hotkey $my_hotkey \
--axon.port $axon_port \
--validator.gate_port $gate_port \
--validator.scoring_backend.host $condense_backend_host \
--validator.scoring_backend.host $condense_backend_port
```

If you want to update the parameters, you can use the following command:
```bash
parameter_name="new_value"
pm2 restart condense_validator --update-env
```