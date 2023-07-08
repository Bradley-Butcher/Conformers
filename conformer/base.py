from dataclasses import dataclass
import itertools
from typing import Callable, Tuple, List

from typing import List, Tuple
import transformers
import torch
from loguru import logger

from typing import List, Tuple

__all__ = ["SWRResult", "ResultStore", "LambdaGrid", "ConformerBase"]

@dataclass
class SWRResult:
    S: int = -1
    S_star: int = -1
    N_C: int = -1

    def get_contribution(self, rho1: float, rho2: float) -> float:
        return rho1 * self.N_C + rho2 * torch.max(self.S - self.S_star, 0) / self.S

class ResultStore(dict):
    def __init__(self):
        super().__init__()

    def lambda_score(self, lambda_config: Tuple[float], rho1: float, rho2: float) -> float:
        return torch.mean([r.get_contribution(rho1, rho2) for r in self[lambda_config]])

    def get_best_lambda(self, rho1: float, rho2: float) -> Tuple[float]:
        best_lambda = None
        best_score = float("inf")
        for lambda_config in self:
            score = self.lambda_score(lambda_config, rho1, rho2)
            if score < best_score:
                best_score = score
                best_lambda = lambda_config
        return best_lambda

    def add_result(self, lambda_config: Tuple[float], result: SWRResult):
        if lambda_config not in self:
            self[lambda_config] = []
        self[lambda_config].append(result)

    def remove_invalid(self, lambda_config):
        del self[lambda_config]

class LambdaGrid:
    def __init__(self, lambda_vals: List[torch.tensor]):
        self.lambda_vals = lambda_vals
        self.grid = self._create_grid(lambda_vals)

    def _create_grid(self, lambda_vals: List[torch.tensor]) -> List[torch.tensor]:
        """Create a grid of lambda values."""
        combinations = list(itertools.product(*lambda_vals))
        return [torch.tensor(c) for c in combinations]
    
    def __getitem__(self, key):
        return self.grid[key]

    def __iter__(self):
        return iter(self.grid)
    
    def __len__(self):
        return len(self.grid)

class ConformerBase:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
    ):  
        if isinstance(model, str):
            self.model = transformers.AutoModel.from_pretrained(model)
        else:
            self.model = model
        if isinstance(tokenizer, str):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.admission_function = None
        self.rejection_functions = []
        self.group_confidence_function = None
        self.lambda_vals = []
        self.func_lambda_map = {}


    def set_admission_function(self, func: Callable) -> None:
        """
        This is a placeholder for the admission function A_i(y). 
        This function should take a tensor y of model predictions and return a tensor of the same shape, 
        indicating whether each prediction is admissible (1) or not (0).
        """
        if self.admission_function is not None:
            logger.warning("Overwriting admission function.")
        self.admission_function = func

    def set_group_confidence_function(self, func: Callable, lambda_vals: torch.tensor) -> None:
        """
        Actually the set-based confidence function but 'set_set' sounds weird.
        """
        self.group_confidence_function = func
        self.lambda_vals.append(lambda_vals)
        self.func_lambda_map[func.__name__] = len(self.lambda_vals) - 1

    def add_rejection_function(
        self, 
        func: Callable, 
        lambda_vals: torch.tensor
    ) -> None:
        assert isinstance(func, Callable), "Rejection function must be callable."
        assert isinstance(lambda_vals, torch.Tensor), "Lambda values must be a tensor."
        self.rejection_functions.append(func)
        self.lambda_vals.append(lambda_vals)
        self.func_lambda_map[func.__name__] = len(self.lambda_vals) - 1