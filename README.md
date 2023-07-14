# Conformers

This is an unofficial implementation of the paper [Conformal Language Modelling](https://arxiv.org/abs/2306.10193).
I found the paper interesting and wanted to play around with it.
Still in a very early state - the only rigorous, statistical guarantee currently is that there are bugs and misunderstandings.
Please excuse the state of the current code - I'll clean it up I promise!

## Status

- [ ] Initial implementation done
- [ ] Tests
- [ ] Pareto Testing procedure (Rather than the current grid search + Bonferroni combo)
- [ ] Component selection
- [ ] PyPI package
- [ ] Experiments

## Changes from the paper

- Sampling is no longer greedy - authors claim to use greedy sampling (default transformer sampling), but this will result in the same output for all samples.
- The selection of the admission function, admission function threshold, and epsilon appears to be very sensitive. In paper the authors select task-dependent admission function thresholds (which I assume they derived experimentally) and try different values of epsilon. In this implementation I will attempt to introduce more generic admission functions.


## Installation

No PyPI package is available yet. To install, clone the repository and run

```bash
pip install poetry
poetry install
```

## Usage

The Python API is not yet set in stone, but the aim is to make it easy to experiment with different admission, group confidence, and rejection functions.
Potentially some quite interesting combinations with the recent CFG language model paper.
Below is an example with GPT2.

```python
from conformer import Calibrator, Sampler, Components
import torch
from random import randint


x =[
    "What is the capital of France?",
    "Which prime-minster of the UK was the biggest nob?",
] 

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

calibrator = Calibrator(
    model=model,
    tokenizer=tokenizer,
    calibration_prompts=x,
)

calibrator.set_admission_function(Components.admission.debug)
calibrator.set_group_confidence_function(Components.group_confidence.debug, torch.tensor([0.1, 0.5, 1]))
calibrator.add_rejection_function(Components.rejection.debug, torch.tensor([0.1, 0.5, 1]))
calibrator.set_FWER(Components.FWER.debug)

lambdaz = calibrator.search()

sampler = Sampler.from_calibrator(calibrator)

sampler.sample_with_rejection("What is the capital of France?")
```

This uses some of the built-in admission/gf/fwer/rejection functions. Can also just use your own function, e.g.:

```python
calibrator.set_group_confidence_function(lambda x: x > 0.5, torch.tensor([0.1, 0.5, 1]))
```