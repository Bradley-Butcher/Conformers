from functools import partial
from scipy.stats import binom
from typing import Callable, List, Optional
import transformers
import torch
from loguru import logger
from tqdm import tqdm
from conformer.base import *
from conformer.base import CElement

class Calibrator(ConformerBase):
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        calibration_prompts: List[str],
        calibration_targets: Optional[List[str]] = None,
        max_calibration_input_length: int = 128,
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
            {"text": c_t, "tokens": self.tok_func(c_t)}
            for c_t in self.calibration_targets
        ] if calibration_targets else None
        logger.info(f"Initialized Conformer with delta={delta} and epsilon={epsilon}.")
        logger.info(f"Precomputing {len(calibration_prompts)} calibration prompts.")
        
        # Caching: Store lambda results in class instance
        self.lambda_results = ResultStore()
        self.calib_store = self._precompute_calibration()

    def set_FWER(self, fwer_algorithm: Callable):
        self.fwer_algorithm = fwer_algorithm

    def _binomial_pval(self, emp_risk: float, n: int) -> float:
        return binom.cdf(n * emp_risk, n, self.epsilon)

    def _is_valid(self, lambda_val: torch.tensor) -> bool:
        e_risk = self._empirical_risk(lambda_val)
        p_val = self._binomial_pval(e_risk, len(self.calibration_prompts))
        return self.fwer_algorithm(p_val, len(lambda_val)) <= self.delta

    def _precompute_calibration(self) -> List[dict]:
        precomputed = {}
        logger.info("Calculating calibration set.")
        self.model.eval()
        with torch.no_grad():
            for prompt in tqdm(self.calibration_prompts, desc="Precomputing calibration set."):
                x = self.tok_func(prompt).to(self.model.device)
                output = self.model.generate(
                    **x, 
                    num_beams=self.samples_per_prompt, 
                    num_return_sequences=self.samples_per_prompt, 
                    max_length=self.max_calibration_output_length, 
                    return_dict_in_generate=True, 
                    output_scores=True, 
                    early_stopping=True
                )
                transition_scores = self.model.compute_transition_scores(
                    output.sequences, 
                    output.scores, 
                    output.beam_indices, 
                    normalize_logits=False
                )
                precomputed[prompt] = [
                    CElement(
                        prompt=prompt,
                        prompt_tokens=x.input_ids[0].detach().cpu(),
                        response=self.tokenizer.decode(output.sequences[i], skip_special_tokens=True),
                        sequence_score=output.sequences_scores[i].detach().cpu(),
                        transition_scores=transition_scores[i].detach().cpu(),
                        response_tokens=output.sequences[i].detach().cpu(),
                    ) for i in range(output.sequences.size(0))]
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
            for reject_func in self.rejection_functions:
                lambda_idx = self.func_lambda_map[reject_func.__name__]
                try:
                    if reject_func(x=prompt, y=y_k, c=C_set, l=lambda_vector[lambda_idx]):
                        break
                except TypeError:
                    raise TypeError(f"Quality estimator function {reject_func.__name__} must take the form f(x, y, c, lambda)")
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
            losses[i] = int(any(self.admission_function(
                self.calibration_prompts[i], 
                C_i, 
                self.calibration_targets[i]) 
                for C_i in C_set
                )
            )
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
