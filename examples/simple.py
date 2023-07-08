
from conformer import Estimator, Sampler
from conformer.fwer import bonferroni_correction
import torch
from random import randint


x =[
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "How many people live in France?",
    "How many people live in Germany?",
    "Which country has the most people?",
    "Which country has the least people?",
    "What is the capital of Spain?",
] 

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

calibrator = Estimator(
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