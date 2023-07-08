from scipy.stats import binom
from typing import Callable, Tuple, List

from typing import List, Tuple
import transformers
import torch
from loguru import logger

from tqdm import tqdm

from typing import List, Tuple

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
        """
        Initialize ConformalModel class. 
        In a complete version of this class, you'd likely need to initialize model parameters, etc.
        """
        super().__init__(model, tokenizer)
        
        self.delta = delta
        self.epsilon = epsilon
        self.func_lambda_map = {}
        self.calibration_prompts = calibration_prompts
        self.rho1 = rho1
        self.rho2 = rho2
        self.k_max = samples_per_prompt

        logger.info(f"Initialized Conformer with delta={delta} and epsilon={epsilon}.")
        logger.info(f"Precomputing {len(calibration_prompts)} calibration prompts.")
        self.calib_output = self._precompute_calibration()
        self.lambda_results = ResultStore()

    def set_FWER(self, fwer_algorithm: Callable):
        """Set the FWER algorithm used for testing hypotheses."""
        self.fwer_algorithm = fwer_algorithm

    def _binomial_pval(self, emp_risk: float, n: int) -> float:
        """Calculate binomial tail bound p-value based on empirical risk, n, and epsilon."""
        return binom.cdf(n * emp_risk, n, self.epsilon)

    def _is_valid(self, lambda_val: torch.tensor) -> bool:
        """Check if a given lambda value is valid."""
        e_risk = self._empirical_risk(lambda_val)
        p_val = self._binomial_pval(e_risk, len(self.calibration_prompts))
        return self.fwer_algorithm(p_val, len(lambda_val)) <= self.delta

    def _precompute_calibration(self, max_length: int = 50) -> Tuple[torch.tensor, torch.tensor]:
        """
        Precompute the calibration set. Contains the dict of model outputs for each calibration prompt.
        Knock yourself out.
        """
        precomputed = []
        logger.info("Calculating calibration set.")
        self.model.eval()
        with torch.no_grad():
            for prompt in tqdm(self.calibration_prompts, total=len(self.calibration_prompts), desc="Precomputing calibration set."):
                x = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.model.device)
                output = self.model.generate(**x, num_beams=self.k_max, num_return_sequences=self.k_max, max_length=max_length, return_dict_in_generate=True, output_scores=True, early_stopping=True)
                transition_scores = self.model.compute_transition_scores(
                    output.sequences, output.scores, output.beam_indices, normalize_logits=False
                )
                for i in range(output.sequences.size(0)):
                    precomputed.append({
                        "prompt": prompt,
                        "response": self.tokenizer.decode(output.sequences[i], skip_special_tokens=True),
                        "sequence_prob": output.sequences_scores[i],
                        "transition_scores": transition_scores[i],
                    }
                    )
        return precomputed

    def _sample_with_rejection(
        self, 
        x: torch.Tensor, 
        lamda_input_vector: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Implements the conformal sampling with rejection algorithm.

        Args:
            x (torch.Tensor): Input prompt tensor of shape (d,), where d is the dimensionality of the prompt.
            F (Callable): Set-based confidence function.
            lambda_ (torch.Tensor): Threshold configuration tensor of shape (3,) for quality, similarity, and confidence thresholds.
            k_max (int): Maximum number of sampling attempts.

        Returns:
            torch.Tensor: Output set tensor of accepted samples, shape depends on specific model and problem.
        """
        # Ensure the reject
        assert self.rejection_functions is not None, "Quality estimator function not set."
        assert self.group_confidence_function is not None, "Group confidence function not set."

        result = SWRResult()
        result.S = self.k_max
        C_lambda = []
        
        # Try to sample k_max times
        group_conf_idx = self.func_lambda_map[self.group_confidence_function.__name__]
        for i in range(self.k_max):
            # Sample a new response y_k
            y_k = self.calib_output[x * self.k_max + i]
            # Reject if any rejection functions proc
            for reject_func in self.rejection_functions:
                lambda_idx = self.func_lambda_map[reject_func.__name__]
                if reject_func(self.calibration_prompts[x], y_k, lamda_input_vector[lambda_idx]):
                    continue

            # Add the new response to the output set
            if result.S_star < 0:
                result.S_star = i + 1

            C_lambda.append(y_k)

            # Check if we are confident enough to stop
            if self.group_confidence_function(C_lambda) > lamda_input_vector[group_conf_idx]:
                result.S = i + 1
                break
        result.S = len(C_lambda)
        self.lambda_results.add_result(lamda_input_vector, result)
        return C_lambda

    def _empirical_risk(
        self, 
        lamda_input_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the empirical risk for given input tensor X and parameter configurations lambdas.
        
        Args:
            X (torch.Tensor): Tensor of input data of shape (n, d), where n is the number of data points 
                and d is the dimensionality of each data point.
            lambdas (torch.Tensor): Tensor of shape (m, p), where m is the number of parameter configurations 
                and p is the number of parameters in each configuration.
        
        Returns:
            torch.Tensor: Tensor of shape (m,) containing the empirical risk for each parameter configuration.
        """
        assert self.admission_function is not None, "Admission function not set."
        # Number of data points
        n = len(self.calibration_prompts)
        
        # Initialize tensor to store loss values
        losses = torch.zeros(n)

        for x_idx in range(n):
            
            # Generate the conformal set C_lambda(X_i)
            C_set = self._sample_with_rejection(x_idx, lamda_input_vector)
            
            # Calculate the loss L_i(lambda)
            # Indicator function: 1 if there doesn't exist an admissible y in C_lambda(X_i), 0 otherwise
            L_i = int(any([self.admission_function(self.calibration_prompts[x_idx], C_i) for C_i in C_set]))
            
            # Store the loss
            losses[x_idx] = L_i
        
        # Calculate and return the empirical risks (average losses for each configuration)        
        return losses.mean(dim=0)

    def search(self):
        grid = LambdaGrid(self.lambda_vals)
        for lambda_vals in tqdm(grid, total=len(grid), desc="Searching for best lambda"):
            if not self._is_valid(lambda_vals):
                self.lambda_results.remove_invalid(lambda_vals)
            else:
                logger.info(f"Found valid lambda: {lambda_vals}")
        results = self.lambda_results.get_best_lambda(self.rho1, self.rho2)
        if results is None:
            raise ValueError("No valid lambda found - try adjusting the rejection / acceptance / confidence functions, or increasing the number of samples.")
        return results