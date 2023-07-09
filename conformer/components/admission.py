"""A script containing a few preset admission functions."""
from conformer.components.shared import ComponentBase
from random import random

def rouge_1_score(x: str, c: list, target: dict, threshold: float):
    reference_tokens = set(y.response_tokens.tolist())
    target_tokens = set(target.response_tokens.tolist())
    shared_tokens = reference_tokens.intersection(target_tokens)
    return (len(shared_tokens) / len(reference_tokens)) >= threshold


def random_admission(x, y, c):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    breakpoint()
    return random()

class AdmissionFunction(ComponentBase):
    random: callable = random_admission
    rouge_1: callable = rouge_1_score
