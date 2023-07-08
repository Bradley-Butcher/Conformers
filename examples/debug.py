
from conformer import Calibrator, Sampler, Components
import torch
from random import randint


x =[
    "What is the capital of France?",
    "Which prime-minster of the UK was the biggest twat?",
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

randint(0, 3) == 0

group_conf_lambdas = torch.tensor([0.1, 0.5, 1])
rejection_lambdas = torch.tensor([0.1, 0.5, 1])

calibrator.set_admission_function(Components.admission.debug)

calibrator.set_group_confidence_function(Components.group_confidence.debug, group_conf_lambdas)

calibrator.add_rejection_function(Components.rejection.debug, rejection_lambdas)

calibrator.set_FWER(Components.FWER.debug)

lambdaz = calibrator.search()


# sampler = Sampler.from_calibrator(calibrator)

# sampler.sample_with_rejection("What is the capital of France?")