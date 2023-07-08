"""A script containing a few preset rejection functions."""
from dataclasses import dataclass
import torch

from conformer.components.shared import ComponentBase


def reject_10(x: str, y: dict, lambdas: torch.tensor):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    if len(x) < 10:
        return True
    return False

class RejectionFunction(ComponentBase):
    reject_10: callable = reject_10