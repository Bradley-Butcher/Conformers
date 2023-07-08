from typing import List
import transformers
import torch

from conformer.base import *

class Sampler(ConformerBase):
    """
    A class derived from ConformerBase, designed for sampling predictions from a model.
    """
    
    def __init__(self, model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, lambda_vector: List[float]):
        """
        Initialize with a model, tokenizer, and lambda vector.
        """
        super().__init__(model, tokenizer)
        self.lambda_vector = lambda_vector

    @classmethod
    def from_estimator(cls, estimator):
        """
        Creates an instance of Sampler from an estimator.
        """
        return cls(estimator.model, estimator.tokenizer, estimator.lambda_vals)

    def sample_with_rejection(self, prompt: str, k_max: int) -> torch.Tensor:
        """
        Samples responses to a prompt, with potential rejection based on certain conditions.
        Tries to sample k_max times and stops if the group confidence function 
        exceeds a threshold specified in the lambda vector.
        """
        assert self.rejection_functions, "Quality estimator function not set."
        assert self.group_confidence_function, "Group confidence function not set."

        C_lambda = []  # Initialize an empty output set
        group_conf_idx = self.func_lambda_map[self.group_confidence_function.__name__]
        
        for _ in range(k_max):
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            y_k = self.model(**tokens, return_dict=True)  # Sample a new response y_k

            # Reject if any rejection functions proc
            for reject_func in self.rejection_functions:
                lambda_idx = self.func_lambda_map[reject_func.__name__]
                if reject_func(prompt, y_k, self.lambda_vector[lambda_idx]):
                    break
            else:
                C_lambda.append(y_k)  # Add the new response to the output set if not rejected

                # Check if we are confident enough to stop
                if self.group_confidence_function(C_lambda) > self.lambda_vector[group_conf_idx]:
                    break

        return torch.stack(C_lambda)
