# Conformers

This is an unofficial implementation of the paper [Conformal Language Modelling](https://arxiv.org/abs/2306.10193).
I found the paper interesting and wanted to play around with it.
Still in a very early state - the only rigorous, statistical guarantee currently is that there are bugs and misunderstandings.

## Status

- [ ] Initial implementation
- [ ] Tests
- [ ] Pareto Testing procedure (Rather than the current grid search + Bonferroni combo)
- [ ] PyPI package
- [ ] Experiments

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
from conformer import Calibrator, Sampler
from conformer.fwer import bonferroni_correction

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

def test_rejection(x, y, l):
    return len(x) < l

calibrator.set_admission_function(lambda x, y:  randint(0, 3) == 0)
calibrator.set_group_confidence_function(lambda x: len(x), torch.tensor([0.1, 0.5, 1]))
calibrator.add_rejection_function(lambda x, y, l: len(x) > l, torch.tensor([1, 2, 5]))
calibrator.add_rejection_function(test_rejection, torch.tensor([0.1, 0.5, 1]))
calibrator.set_FWER(bonferroni_correction)

lambdaz = calibrator.search()

sampler = Sampler.from_calibrator(calibrator)

sampler.sample_with_rejection("What is the capital of France?")
```