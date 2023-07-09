
from conformer import Calibrator, Sampler, Components
import torch
from random import randint


x =[
    "What is the capital of France?",
    "Which prime-minster of the UK was the biggest twat?",
]

y = [
    "The capital of France is Paris",
    "The biggest twat of a prime-minister was Boris Johnson."
]

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

calibrator = Calibrator(
    model=model,
    tokenizer=tokenizer,
    calibration_prompts=x,
    calibration_targets=y
)

group_conf_lambdas = torch.tensor([0.1, 0.5, 1])
rejection_lambdas = torch.tensor([0.1, 0.5, 1])

calibrator.set_admission_function(
    func=Components.admission.rouge_1, 
    threshold=0.4
)

calibrator.set_group_confidence_function(
    Components.group_confidence.sum_ngll, 
    group_conf_lambdas
)

calibrator.add_rejection_function(
    Components.rejection.rouge_1, 
    rejection_lambdas
)

calibrator.add_rejection_function(
    Components.rejection.ngll, 
    rejection_lambdas
)

calibrator.set_FWER(
    fwer_algorithm=Components.FWER.bonferroni_correction
)

lambdaz = calibrator.search()


# sampler = Sampler.from_calibrator(calibrator)

# sampler.sample_with_rejection("What is the capital of France?")