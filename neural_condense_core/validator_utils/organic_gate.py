from fastapi import FastAPI, Depends
import pydantic
import asyncio
import bittensor as bt
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import logging
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from ..constants import ORGANIC_CLIENT_URL, TIER_CONFIG
from .. import __spec_version__
from ..protocol import TextCompressProtocol
from ..validator_utils import MinerManager

LOGGER = logging.getLogger("organic_gate")


class OrganicRequest(pydantic.BaseModel):
    text_to_compress: str
    model_name: str
    tier: str
    uid: int = -1


class OrganicResponse(pydantic.BaseModel):
    compressed_tokens: list[list[float]]


class RegisterSynapse(bt.Synapse):
    port: int


class OrganicGate:
    def __init__(
        self,
        miner_manager: MinerManager,
        wallet: bt.wallet,
        config: bt.config,
        metagraph,
    ):
        self.metagraph = metagraph
        self.miner_manager = miner_manager
        self.wallet = wallet
        self.config = config
        self.get_credentials()
        self.dendrite = bt.dendrite(wallet=wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/forward",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=[ORGANIC_CLIENT_URL, "localhost"]
        )
        self.loop = asyncio.get_event_loop()
        self.client_axon: bt.AxonInfo = None
        self.start_server()

    def register_to_client(self):
        synapse = RegisterSynapse(port=self.config.validator.gate_port)
        self.call(self.dendrite, ORGANIC_CLIENT_URL, synapse)

    async def forward(self, request: OrganicRequest):
        synapse = TextCompressProtocol(
            context=request.text_to_compress,
        )
        if request.uid != -1:
            targeted_uid = request.uid
        else:
            for uid, counter in self.miner_manager.serving_counter[
                request.tier
            ].items():
                if counter.increment():
                    targeted_uid = uid
                    break

        target_axon = self.metagraph.axons[targeted_uid]

        response: TextCompressProtocol = await self.dendrite.forward(
            axons=[target_axon],
            synapse=synapse,
            timeout=TIER_CONFIG[request.tier]["timeout"],
        )
        return OrganicResponse(compressed_tokens=response.compressed_tokens)

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run,
            self.app,
            host="0.0.0.0",
            port=self.config.validator.gate_port,
        )

    async def get_self(self):
        return self

    async def call(
        self,
        dendrite: bt.dendrite,
        url: str,
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12.0,
    ) -> bt.Synapse:
        """
        Customized call method to send Synapse-like requests to the Organic Client Server.

        Args:
            dendrite (bt.Dendrite): The Dendrite object to send the request.
            url (str): The URL of the Organic Client Server.
            synapse (bt.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new :func:`bt.Synapse` instance.
            timeout (float, optional): Maximum duration to wait for a response from the Axon in seconds. Defaults to ``12.0``.

        Returns:
            bt.Synapse: The Synapse object, updated with the response data from the Axon.
        """

        request_name = synapse.__class__.__name__
        url = f"{ORGANIC_CLIENT_URL}/{request_name}"
        target_axon = bt.AxonInfo(
            ip="0.0.0.0",
            port="8080",
            hotkey="unknown",
            coldkey="unknown",
            version=__spec_version__,
            ip_type="ipv4",
        )
        synapse = dendrite.preprocess_synapse_for_request(target_axon, synapse, timeout)
        try:
            async with (await dendrite.session).post(
                url,
                headers=synapse.to_headers(),
                json=synapse.model_dump(),
                timeout=timeout,
            ) as response:
                json_response = await response.json()
                dendrite.process_server_response(response, json_response, synapse)
        except Exception as e:
            LOGGER.error(f"Failed to send request: {e}")
        return synapse
