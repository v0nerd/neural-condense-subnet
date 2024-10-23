import neural_condense_core as ncc
import httpx


class Miner(ncc.BaseMiner):
    def __init__(self):
        super().__init__()
        self.forward_fns.append(self.forward_text_compress)
        self.blacklist_fns.append(self.blacklist_fn)
        self.rate_limits = ncc.build_rate_limit(self.metagraph, self.config)

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
                f"http://{self.config.backend_host}:{self.config.backend_port}/forward_text_compress",
                json=payload,
                timeout=synapse.timeout,
            )
            response = response.json()
            compressed_tokens = response["compressed_tokens"]
            synapse.compressed_tokens = compressed_tokens
        return synapse

    def blacklist_fn(self, synapse: ncc.TextCompressProtocol):
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
        if stake < self.config.miner.min_stake:
            return True, "Stake too low."
        allowed = self.rate_limits[uid].increment()
        if not allowed:
            return True, "Rate limit exceeded."
        return False, ""


if __name__ == "__main__":
    miner = Miner()
    miner.run()
