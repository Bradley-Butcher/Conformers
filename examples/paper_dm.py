
from conformer import Calibrator, Sampler, Components
import torch
from random import randint
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset


dataset = load_dataset("cnn_dailymail", "3.0.0")

x = dataset["train"][:50]["article"]

# Append to each x ". Summary: "
x = [x_i[:] + ". Write a scientific version of the article, with lots of scientific language: " for x_i in x]
y = dataset["train"][:50]["highlights"]

model_name = "psmathur/orca_mini_3b"
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
).cuda()
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

calibrator = Calibrator(
    model=model,
    tokenizer=tokenizer,
    calibration_prompts=x,
    calibration_targets=y,
    # delta=0.2,
    epsilon=0.3,
    calibration_path="calibration_store.pkl",
)

group_conf_lambdas = torch.tensor([0.2, 0.4, 0.6, 0.8, 1])
nll_rej_lambdas = torch.tensor([0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
rouge_rej_lambdas = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.9])

calibrator.set_admission_function(
    func=Components.admission.ppl_prefix(tokenizer, model, "Scientific Article:"),
    threshold=0.5
)

calibrator.set_group_confidence_function(
    Components.group_confidence.sum_ngll, 
    group_conf_lambdas
)

calibrator.add_rejection_function(
    Components.rejection.rouge_1, 
    rouge_rej_lambdas
)

calibrator.add_rejection_function(
    Components.rejection.ngll, 
    nll_rej_lambdas
)

calibrator.set_FWER(
    fwer_algorithm=Components.FWER.none_debug
)

lambdaz = calibrator.search()

breakpoint()

# sampler = Sampler.from_calibrator(calibrator)

# sampler.sample_with_rejection("What is the capital of France?")