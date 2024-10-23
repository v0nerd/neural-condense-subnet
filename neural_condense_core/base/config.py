import bittensor as bt
from argparse import ArgumentParser
from ..constants import TIER_CONFIG


def add_common_config(parser: ArgumentParser):
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    return parser


def add_validator_config(parser: ArgumentParser):
    parser.add_argument(
        "--validator.gate_port",
        type=int,
        default=12345,
        help="The port of the validator gate server.",
    )
    return parser


def add_miner_config(parser: ArgumentParser):
    tier_names = list(TIER_CONFIG.keys())
    parser.add_argument(
        "--miner.backend_host",
        type=str,
        default="localhost",
        help="The host of the backend server.",
    )
    parser.add_argument(
        "--miner.backend_port",
        type=int,
        default=8088,
        help="The port of the backend server.",
    )
    parser.add_argument(
        "--miner.tier",
        choices=tier_names,
    )
    return parser
