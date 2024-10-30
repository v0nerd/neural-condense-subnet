import bittensor as bt
from ..constants import constants


def build_rate_limit(metagraph, config, tier=None):
    S = metagraph.S
    whitelist_uids = [i for i in range(len(S)) if S[i] > constants.MIN_STAKE]
    bt.logging.debug(f"Whitelist uids: {whitelist_uids}")
    selected_tier_config = constants.TIER_CONFIG[tier or config.miner.tier]
    rpe = selected_tier_config.requests_per_epoch
    rpe_per_validator = rpe // len(whitelist_uids)
    rate_limits = {uid: rpe_per_validator for uid in whitelist_uids}
    bt.logging.debug(f"Rate limits: {rate_limits}")
    for uid in range(len(S)):
        if uid not in whitelist_uids:
            rate_limits[uid] = 0
    return rate_limits
