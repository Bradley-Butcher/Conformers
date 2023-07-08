"""A script containing a few preset rejection functions."""
from dataclasses import dataclass
import torch

from conformer.components.shared import ComponentBase

from random import random


def random_reject(x: str, y: dict, l: torch.tensor):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    seed = sum([int(c) for c in x])
    random.seed(seed)
    return random() > l

class RejectionFunction(ComponentBase):
    random: callable = random_reject