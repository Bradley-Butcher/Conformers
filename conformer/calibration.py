from scipy.stats import binom
from typing import Callable, List
import transformers
import torch
from loguru import logger
from tqdm import tqdm
from conformer.base import *

class Calibrator(ConformerBase):
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        calibration_prompts: List[str],
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
        self.rho1 = rho1
        self.rho2 = rho2
        self.samples_per_prompt = samples_per_prompt

        logger.info(f"Initialized Conformer with delta={delta} and epsilon={epsilon}.")
        logger.info(f"Precomputing {len(calibration_prompts)} calibration prompts.")
        
        # Caching: Store lambda results in class instance
        self.lambda_results = ResultStore()
        self.calib_output = self._precompute_calibration()

    def set_FWER(self, fwer_algorithm: Callable):
        self.fwer_algorithm = fwer_algorithm

    def _binomial_pval(self, emp_risk: float, n: int) -> float:
        return binom.cdf(n * emp_risk, n, self.epsilon)

    def _is_valid(self, lambda_val: torch.tensor) -> bool:
        e_risk = self._empirical_risk(lambda_val)
        p_val = self._binomial_pval(e_risk, len(self.calibration_prompts))
        return self.fwer_algorithm(p_val, len(lambda_val)) <= self.delta

    def _precompute_calibration(self, max_length: int = 50) -> List[dict]:
        precomputed = []
        logger.info("Calculating calibration set.")
        self.model.eval()
        with torch.no_grad():
            for prompt in tqdm(self.calibration_prompts, desc="Precomputing calibration set."):
                x = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.model.device)
                output = self.model.generate(**x, num_beams=self.samples_per_prompt, num_return_sequences=self.samples_per_prompt, max_length=max_length, return_dict_in_generate=True, output_scores=True, early_stopping=True)
                transition_scores = self.model.compute_transition_scores(
                    output.sequences, output.scores, output.beam_indices, normalize_logits=False
                )
                precomputed.extend([{
                    "prompt": prompt,
                    "response": self.tokenizer.decode(output.sequences[i], skip_special_tokens=True),
                    "sequence_prob": output.sequences_scores[i].detach().cpu(),
                    "transition_scores": transition_scores[i].detach().cpu(),
                } for i in range(output.sequences.size(0))])
        return precomputed

    def _sample_with_rejection(self, x: int, lambda_vector: torch.Tensor) -> List[dict]:
        assert self.rejection_functions is not None, "Quality estimator function not set."
        assert self.group_confidence_function is not None, "Group confidence function not set."

        result = SWRResult()
        result.S = self.samples_per_prompt
        C_lambda = []

        group_conf_idx = self.func_lambda_map[self.group_confidence_function.__name__]
        for i in range(self.samples_per_prompt):
            y_k = self.calib_output[x * self.samples_per_prompt + i]
            for reject_func in self.rejection_functions:
                lambda_idx = self.func_lambda_map[reject_func.__name__]
                if reject_func(x=self.calibration_prompts[x], y=y_k, l=lambda_vector[lambda_idx]):
                    break
            if result.S_star < 0:
                result.S_star = i + 1
            C_lambda.append(y_k)
            if self.group_confidence_function(C_lambda) > lambda_vector[group_conf_idx]:
                result.S = i + 1
                break
        result.S = len(C_lambda)
        self.lambda_results.add_result(lambda_vector, result)
        return C_lambda

    def _empirical_risk(self, lambda_vector: torch.Tensor) -> torch.Tensor:
        assert self.admission_function is not None, "Admission function not set."
        n = len(self.calibration_prompts)
        losses = torch.zeros(n)
        for x_idx in range(n):
            C_set = self._sample_with_rejection(x_idx, lambda_vector)
            losses[x_idx] = int(any(self.admission_function(self.calibration_prompts[x_idx], C_i) for C_i in C_set))
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
