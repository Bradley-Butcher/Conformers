import transformers
import torch

from conformer.base import *

class Sampler(ConformerBase):
    def __init__(
        self, 
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        lambda_vector
    ):
        super.__init__(model, tokenizer)
        self.lambda_vector = lambda_vector

    @classmethod
    def from_estimator(cls, estimator):
        return cls(
            estimator.model, 
            estimator.tokenizer, 
            estimator.lambda_vals
        )

    def sample_with_rejection(
        self, 
        prompt: str,
        k_max: int,
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
        
        # Initialize an empty output set
        C_lambda = []

        # Try to sample k_max times
        group_conf_idx = self.func_lambda_map[self.group_confidence_function.__name__]
        for i in range(k_max):
            # Sample a new response y_k
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            y_k = self.model(**tokens, return_dict=True)
            # Reject if any rejection functions proc
            for reject_func in self.rejection_functions:
                lambda_idx = self.func_lambda_map[reject_func.__name__]
                if reject_func(prompt, y_k, self.lambda_vector[lambda_idx]):
                    continue

            # Add the new response to the output set
            C_lambda.append(y_k)

            # Check if we are confident enough to stop
            if self.group_confidence_function(C_lambda) > self.lambda_vector[group_conf_idx]:
                break

        return torch.stack(C_lambda)