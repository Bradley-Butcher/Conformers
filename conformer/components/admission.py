"""A script containing a few preset admission functions."""
from functools import cache
import torch
from conformer.components.shared import ComponentBase
from random import random


def setup_rouge_1(threshold: float):
    def rouge_1_score(x: str, c: list, target: dict):
        prompt_len = c.prompt_tokens.size(0)
        just_response = c.response_tokens[prompt_len:]
        reference_tokens = set(just_response.tolist())
        target_tokens = set(target["tokens"].tolist())
        shared_tokens = reference_tokens.intersection(target_tokens)
        return (len(shared_tokens) / len(target_tokens)) > threshold
    return rouge_1_score

def setup_ppl_prefix(tokenizer, model, prefix: str):
    def ppl_prefix(x: str, c: list, target: dict):
        prompt_len = len(c.prompt)
        response_str = c.response[prompt_len:]
        @cache
        def _inner(response: str, prefix_str: str):
            inputs = tokenizer(response, return_tensors="pt").to(model.device)
            inputs_with_prefix = tokenizer(prefix_str + response, return_tensors="pt").to(model.device)
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            loss_with_prefix = model(**inputs_with_prefix, labels=inputs_with_prefix["input_ids"]).loss
            ppl = torch.exp(loss)
            ppl_with_prefix = torch.exp(loss_with_prefix)
            return ppl.detach(), ppl_with_prefix.detach()
        ppl, ppl_with_prefix = _inner(response_str, prefix)
        return ppl_with_prefix > ppl
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

def setup_rouge_l(threshold: float):
    def rouge_l_score(x: str, c: list, target: dict) -> float:
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
    return rouge_l_score


def random_admission(x, y, c):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    breakpoint()
    return random()

class AdmissionFunction(ComponentBase):
    random: callable = random_admission
    rouge_1: callable = setup_rouge_1
    rouge_l: callable = setup_rouge_l
    ppl_prefix: callable = setup_ppl_prefix
