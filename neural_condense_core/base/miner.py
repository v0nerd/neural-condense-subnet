import os
import time
import argparse
import traceback
import bittensor as bt
from .config import add_common_config, add_miner_config
from ..protocol import Metadata


class Miner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.blacklist_fns = [self._blacklist_fn]
        self.forward_fns = [self._forward_metadata]
        self.setup_bittensor_objects()
        self.metadata = {
            "tier": self.config.miner.tier,
        }

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser = add_miner_config(parser)
        parser = add_common_config(parser)
        config = parser.parse_args()
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid_{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                "miner",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Activate Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
        )
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        bt.logging.info("Setting up Bittensor objects.")
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def setup_axon(self):
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)
        bt.logging.info("Attaching forward function to axon.")
        for blacklist_fn, forward_fn in zip(self.blacklist_fns, self.forward_fns):
            bt.logging.info(
                f"Attaching blacklist_fn: {blacklist_fn} and forward_fn: {forward_fn}"
            )
            self.axon.attach(
                forward_fn=forward_fn,
                blacklist_fn=blacklist_fn,
            )
        bt.logging.info(
            f"Serving axon on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Axon: {self.axon}")
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

    def _forward_metadata(self, synapse: Metadata):
        synapse.metadata = self.metadata
        return synapse

    def _blacklist_fn(self, synapse: Metadata):
        return False

    def run(self):
        self.setup_axon()
        bt.logging.info("Starting main loop")
        step = 0
        while True:
            try:
                # Periodically update our knowledge of the network graph.
                if step % 60 == 0:
                    self.metagraph.sync()
                    log = (
                        f"Block: {self.metagraph.block.item()} | "
                        f"Incentive: {self.metagraph.I[self.my_subnet_uid]} | "
                    )
                    bt.logging.info(log)
                step += 1
                time.sleep(10)

            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(f"Miner exception: {e}")
                bt.logging.error(traceback.format_exc())
                continue
