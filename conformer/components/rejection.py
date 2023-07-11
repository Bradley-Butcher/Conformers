"""A script containing a few preset rejection functions."""
from dataclasses import dataclass
import torch
from conformer.base import CElement

from conformer.components.shared import ComponentBase

from random import random

def rouge_1_score(x: str, y: CElement, c: list, l: torch.tensor):
    reference_tokens = set(y.response_tokens.tolist())
    def _inner_rogue(y_i):
        hypothesis_tokens = set(y_i.response_tokens.tolist())
        # Calculate the number of shared tokens
        shared_tokens = reference_tokens.intersection(hypothesis_tokens)
        # Calculate the Rouge-1 score
        return len(shared_tokens) / len(reference_tokens)
    max_rouge = max([_inner_rogue(y_i) for y_i in c]) if c else 0
    return max_rouge >= l

def ngll_score(x: str, y: CElement, c: list, l: torch.tensor):
    return -y.sequence_score >= l

def random_reject(x: str, y: CElement, c: list, l: torch.tensor):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    seed = sum([int(ci) for ci in x])
    random.seed(seed)
    return random() > l

class RejectionFunction(ComponentBase):
    random: callable = random_reject
    rouge_1: callable = rouge_1_score
    ngll: callable = ngll_score