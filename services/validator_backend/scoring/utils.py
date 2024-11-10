import numpy as np


def loss_to_scores(losses: list[float]) -> list[float]:
    return [-loss for loss in losses]
