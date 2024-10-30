import neural_condense_core as ncc
import httpx
from typing import Tuple


class Miner(ncc.BaseMiner):
    def __init__(self):
        super().__init__()
        self.blacklist_fns.append(self.blacklist_fn)
        self.forward_fns.append(self.forward_text_compress)
        self.setup_logging()
        self.rate_limits = {
            uid: ncc.ServingCounter(rate_limit)
            for uid, rate_limit in ncc.build_rate_limit(
                self.metagraph, self.config
            ).items()
        }

    async def forward_text_compress(
        self, synapse: ncc.TextCompressProtocol
    ) -> ncc.TextCompressProtocol:
        r"""
        Forward function for the Text-Compress task.
        Args:
            synapse (TextCompressProtocol): The synapse containing the text to compress.
        Returns:
            synapse (TextCompressProtocol): The synapse containing the compressed tokens.
        """
        payload = synapse.get_miner_payload()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.config.miner.backend_host}:{self.config.miner.backend_port}/forward_text_compress",
                json=payload,
                timeout=synapse.timeout,
            )
            response = response.json()
            compressed_tokens = response["compressed_tokens"]
            synapse.compressed_tokens = compressed_tokens
        return synapse

    def blacklist_fn(self, synapse: ncc.TextCompressProtocol) -> Tuple[bool, str]:
        r"""
        Blacklist function for the Text-Compress task.
        Args:
            synapse (TextCompressProtocol): The synapse containing the text to compress.
        Returns:
            bool: Whether to blacklist the synapse.
            reason (str): The reason for blacklisting the synapse.
        """
        hotkey = synapse.dendrite.hotkey
        uid = self.metagraph.hotkeys.index(hotkey)
        stake = self.metagraph.S[uid]
        if stake < ncc.constants.MIN_STAKE:
            return True, "Stake too low."
        allowed = self.rate_limits[uid].increment()
        if not allowed:
            return True, "Rate limit exceeded."
        return False, ""


if __name__ == "__main__":
    miner = Miner()
    miner.run()
