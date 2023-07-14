from functools import partial
import pickle
from scipy.stats import binom
from typing import Callable, List, Optional
import transformers
import torch
from loguru import logger
from tqdm import tqdm
from conformer.base import *
from conformer.base import CElement
import os

class Calibrator(ConformerBase):
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        calibration_prompts: List[str],
        calibration_path: Optional[str] = None,
        calibration_targets: Optional[List[str]] = None,
        max_calibration_input_length: int = 1536,
        max_calibration_output_length: int = 256,
        samples_per_prompt: int = 5,
        delta: float = 0.05,
        epsilon: float = 0.05,
        rho1: float = 0.5,
        rho2: float = 0.5,
    ):
        super().__init__(model, tokenizer)
        self.delta = delta
        self.epsilon = epsilon
        self.calibration_prompts = calibration_prompts
        self.calibration_targets = calibration_targets
        self.rho1 = rho1
        self.rho2 = rho2
        self.samples_per_prompt = samples_per_prompt

        self.max_calibration_output_length = max_calibration_output_length

        self.tok_func = partial(self.tokenizer, return_tensors="pt", padding=True, truncation=True, max_length=max_calibration_input_length)
        self.calibration_targets = [
            {"text": c_t, "tokens": self.tok_func(c_t)["input_ids"][0]}
            for c_t in self.calibration_targets
        ] if calibration_targets else None
        logger.info(f"Initialized Conformer with delta={delta} and epsilon={epsilon}.")
        logger.info(f"Precomputing {len(calibration_prompts)} calibration prompts.")
        
        # Caching: Store lambda results in class instance
        self.lambda_results = ResultStore()
        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path, "rb") as f:
                self.calib_store = pickle.load(f)
        else:
            self.calib_store = self._precompute_calibration()
            if calibration_path:
                with open(calibration_path, "wb") as f:
                    pickle.dump(self.calib_store, f)

    def save_calibration(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.calib_store, f)


    def set_FWER(self, fwer_algorithm: Callable):
        self.fwer_algorithm = fwer_algorithm

    def _binomial_pval(self, emp_risk: float, n: int) -> float:
        return binom.cdf(n * emp_risk, n, self.epsilon)

    def _is_valid(self, lambda_val: torch.tensor) -> bool:
        e_risk = self._empirical_risk(lambda_val)
        logger.info(f"Empirical risk: {e_risk}")
        p_val = self._binomial_pval(e_risk, len(self.calibration_prompts))
        if self.fwer_algorithm(p_val, len(lambda_val)) <= self.delta:
            return True
        else:
            return False

    def _precompute_calibration(self) -> List[dict]:
        precomputed = {}
        logger.info("Calculating calibration set.")
        self.model.eval()
        with torch.no_grad():
            for prompt in tqdm(self.calibration_prompts, desc="Precomputing calibration set."):
                x = self.tok_func(prompt).to(self.model.device)
                output = self.model.generate(
                    **x, 
                    max_new_tokens=self.max_calibration_output_length, 
                    num_return_sequences=self.samples_per_prompt,
                    return_dict_in_generate=True, 
                    output_scores=True,
                    do_sample=True, 
                    top_k=50, 
                    top_p=0.95, 
                )
                elements = []
                for i in range(len(output.sequences)):
                    prompt_n = len(x["input_ids"][0])
                    scores = tuple([output.scores[j][i].unsqueeze(0) for j in range(len(output.scores))])
                    transitions = self.model.compute_transition_scores(output.sequences[i][prompt_n:].unsqueeze(0),scores,normalize_logits=True)
                    transitions = torch.nan_to_num(transitions, neginf=0.0)
                    elements.append(CElement(
                            prompt=prompt,
                            prompt_tokens=x.input_ids[0].detach().cpu(),
                            response=self.tokenizer.decode(output.sequences[i], skip_special_tokens=True),
                            sequence_score=transitions.sum().detach().cpu() / transitions.size(1),
                            transition_scores=transitions.detach().cpu(),
                            response_tokens=output.sequences[i].detach().cpu(),
                    ))
                precomputed[prompt] = elements
        return precomputed

    def _sample_with_rejection(self, prompt: str, lambda_vector: torch.Tensor) -> List[dict]:
        assert self.rejection_functions is not None, "Quality estimator function not set."
        assert self.group_confidence_function is not None, "Group confidence function not set."

        result = SWRResult()
        result.S = self.samples_per_prompt
        C_set = []
        group_conf_idx = self.func_lambda_map[self.group_confidence_function.__name__]
        for i in range(self.samples_per_prompt):
            y_k = self.calib_store[prompt][i]
            break_loop = False
            for reject_func in self.rejection_functions:
                lambda_idx = self.func_lambda_map[reject_func.__name__]
                try:
                    if reject_func(x=prompt, y=y_k, c=C_set, l=lambda_vector[lambda_idx]):
                        break_loop = True
                        break
                except TypeError:
                    raise TypeError(f"Quality estimator function {reject_func.__name__} must take the form f(x, y, c, lambda)")
            if break_loop:
                continue
            if result.S_star < 0:
                result.S_star = i + 1
            C_set.append(y_k)
            try:
                if self.group_confidence_function(prompt, C_set, lambda_vector[group_conf_idx]):
                    result.S = i + 1
                    break
            except TypeError:
                raise TypeError(f"Group confidence function {self.group_confidence_function.__name__} must take the form f(prompt, C_set, lambda)")
        result.S = len(C_set)
        self.lambda_results.add_result(lambda_vector, result)
        return C_set

    def _empirical_risk(self, lambda_vector: torch.Tensor) -> torch.Tensor:
        assert self.admission_function is not None, "Admission function not set."
        n = len(self.calibration_prompts)
        losses = torch.zeros(n)
        for i, prompt in enumerate(self.calibration_prompts):
            C_set = self._sample_with_rejection(prompt, lambda_vector)
            # losses[i] = int(any(self.admission_function(self.calibration_prompts[i], C_i, self.calibration_targets[i]) for C_i in C_set))
            admitted = []
            for C_i in C_set:
                if self.admission_function(prompt, C_i, self.calibration_targets[i]):
                    admitted.append(C_i)
            if not admitted:
                losses[i] = 1
        return losses.mean()

    def search(self):
        grid = LambdaGrid(self.lambda_vals)
        for lambda_vals in tqdm(grid, desc="Searching for best lambda"):
            if not self._is_valid(lambda_vals):
                self.lambda_results.remove_invalid(lambda_vals)
            else:
                logger.info(f"Found valid lambda: {lambda_vals}")
        results = self.lambda_results.get_best_lambda(self.rho1, self.rho2)
        if results is None:
            raise ValueError("No valid lambda found - try adjusting the rejection / acceptance / confidence functions, or increasing the number of samples.")
        return results
