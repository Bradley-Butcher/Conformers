"""A script containing a few preset admission functions."""
from conformer.components.shared import ComponentBase
from random import random

def rouge_1_score(x: str, c: list, target: dict, threshold: float):
    prompt_len = c.prompt_tokens.size(0)
    just_response = c.response_tokens[prompt_len:]
    reference_tokens = set(just_response.tolist())
    target_tokens = set(target["tokens"].tolist())
    shared_tokens = reference_tokens.intersection(target_tokens)
    return (len(shared_tokens) / len(target_tokens)) > threshold


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
