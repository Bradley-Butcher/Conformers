"""A script containing a few preset group confidence functions."""

import torch
from conformer.components.shared import ComponentBase

from random import random

def random_gc(x: str, c: list, l: torch.tensor):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    breakpoint()
    return random()

def set_size(x: str, c: list, l: torch.tensor):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    return len(c) >= l

def max_ngll(x: str, c: list, l: torch.tensor):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    return max([ci.sequence_score for ci in c]) <= l

def sum_ngll(x: str, c: list, l: torch.tensor):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    return sum([ci.sequence_score for ci in c]) <= l

class GroupConfidenceFunction(ComponentBase):
    random: callable = random_gc
    set_size: callable = set_size
    max_ngll: callable = max_ngll
    sum_ngll: callable = sum_ngll