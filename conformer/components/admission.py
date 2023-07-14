"""A script containing a few preset admission functions."""
import torch
from conformer.components.shared import ComponentBase
from random import random

def rouge_1_score(x: str, c: list, target: dict, threshold: float):
    prompt_len = c.prompt_tokens.size(0)
    just_response = c.response_tokens[prompt_len:]
    reference_tokens = set(just_response.tolist())
    target_tokens = set(target["tokens"].tolist())
    shared_tokens = reference_tokens.intersection(target_tokens)
    return (len(shared_tokens) / len(target_tokens)) > threshold

def setup_ppl_prefix(tokenizer, model, prefix: str):
    def ppl_prefix(x: str, c: list, target: dict, threshold: float):
        prompt_len = c.prompt.size(0)
        response = c.response[prompt_len:]
        inputs = tokenizer(response, return_tensors="pt")
        inputs_with_prefix = tokenizer(prefix + response, return_tensors="pt")
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        loss_with_prefix = model(**inputs_with_prefix, labels=inputs_with_prefix["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl_with_prefix = torch.exp(loss_with_prefix)
        return (ppl / ppl_with_prefix) < threshold
    return ppl_prefix

from typing import List
from numpy import array

def lcs(x: List[int], y: List[int]) -> int:
    m = len(x)
    n = len(y)
    L = array([[0] * (n + 1) for _ in range(m + 1)])

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def rouge_l_score(x: str, c: list, target: dict, threshold: float) -> float:
    prompt_len = c.prompt_tokens.size(0)
    just_response = c.response_tokens[prompt_len:]
    reference_tokens = just_response.tolist()
    target_tokens = target["tokens"].tolist()

    lcs_length = lcs(reference_tokens, target_tokens)

    # Prevent division by zero
    if len(target_tokens) == 0 or len(reference_tokens) == 0:
        return 0.0

    precision = lcs_length / len(reference_tokens)
    recall = lcs_length / len(target_tokens)

    # Harmonic mean of precision and recall
    if precision + recall != 0:
        rouge_l = 2 * precision * recall / (precision + recall)
    else:
        rouge_l = 0.0

    return rouge_l > threshold


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
    rouge_l: callable = rouge_l_score
    ppl_prefix: callable = setup_ppl_prefix
