from dataclasses import dataclass
from functools import partial
import itertools
from typing import Callable, Optional, Tuple, List

from typing import List, Tuple
import transformers
import torch
from loguru import logger

from typing import List, Tuple

__all__ = ["SWRResult", "ResultStore", "LambdaGrid", "ConformerBase"]


@dataclass
class CElement:
    prompt: str
    prompt_tokens: torch.tensor
    response: str
    response_tokens: torch.tensor
    sequence_score: float
    transition_scores: float

@dataclass
class SWRResult:
    """
    Data class for holding results of SWR.
    """
    S: int = -1
    S_star: int = -1
    N_C: int = -1

    def get_contribution(self, rho1: float, rho2: float) -> float:
        """
        Get the contribution of a SWR result.
        """
        return rho1 * self.N_C + rho2 * max(self.S - self.S_star, 0) / self.S


class ResultStore(dict):
    """
    Class for storing and managing SWR results.
    """

    def lambda_score(self, lambda_config: Tuple[float], rho1: float, rho2: float) -> float:
        """
        Compute the score for a specific lambda configuration.
        """
        return torch.mean([r.get_contribution(rho1, rho2) for r in self[lambda_config]])

    def get_best_lambda(self, rho1: float, rho2: float) -> Tuple[float]:
        """
        Get the best lambda configuration based on the scores.
        """
        scores = {lambda_config: self.lambda_score(lambda_config, rho1, rho2) for lambda_config in self}
        best_lambda = min(scores, key=scores.get)
        return best_lambda

    def add_result(self, lambda_config: Tuple[float], result: SWRResult):
        """
        Add a SWR result for a specific lambda configuration.
        """
        self.setdefault(lambda_config, []).append(result)

    def remove_invalid(self, lambda_config):
        """
        Remove an invalid lambda configuration.
        """
        self.pop(lambda_config, None)


class LambdaGrid:
    """
    Class for generating a grid of lambda values.
    """

    def __init__(self, lambda_vals: List[torch.tensor]):
        """
        Initialize with a list of lambda values.
        """
        self.lambda_vals = lambda_vals
        self.grid = self._create_grid(lambda_vals)

    def _create_grid(self, lambda_vals: List[torch.tensor]) -> List[torch.tensor]:
        """
        Private method to create a grid of lambda values.
        """
        combinations = list(itertools.product(*lambda_vals))
        return [torch.tensor(c) for c in combinations]
    
    def __getitem__(self, key):
        """
        Enable indexing for LambdaGrid.
        """
        return self.grid[key]

    def __iter__(self):
        """
        Enable iteration for LambdaGrid.
        """
        return iter(self.grid)
    
    def __len__(self):
        """
        Enable len() function for LambdaGrid.
        """
        return len(self.grid)



class ConformerBase:
    """
    Base class for Conformer which holds a pretrained model and tokenizer.
    Also includes functions for conformity measures.
    """

    def __init__(
            self, 
            model: transformers.PreTrainedModel, 
            tokenizer: transformers.PreTrainedTokenizer):
        """
        Initialize with a model and tokenizer.
        """
        self.model = transformers.AutoModel.from_pretrained(model) if isinstance(model, str) else model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.admission_function = None
        self.rejection_functions = []
        self.group_confidence_function = None
        self.lambda_vals = []
        self.func_lambda_map = {}

    def set_admission_function(
        self, 
        func: Callable,
        threshold: Optional[float] = None,
                               
    ) -> None:
        """
        Set the admission function for Conformer.
        """
        if self.admission_function is not None:
            logger.warning("Overwriting admission function.")
        if threshold is not None:
            self.admission_function = partial(func, threshold=threshold)
        else:
            self.admission_function = func

    def set_group_confidence_function(self, func: Callable, lambda_vals: torch.tensor) -> None:
        """
        Set the group confidence function for Conformer.
        """
        self.group_confidence_function = func
        self.lambda_vals.append(lambda_vals)
        self.func_lambda_map[func.__name__] = len(self.lambda_vals) - 1

    def add_rejection_function(self, func: Callable, lambda_vals: torch.tensor) -> None:
        """
        Add a rejection function to the Conformer.
        """
        assert isinstance(func, Callable), "Rejection function must be callable."
        assert isinstance(lambda_vals, torch.Tensor), "Lambda values must be a tensor."
        self.rejection_functions.append(func)
        self.lambda_vals.append(lambda_vals)
        self.func_lambda_map[func.__name__] = len(self.lambda_vals) - 1
