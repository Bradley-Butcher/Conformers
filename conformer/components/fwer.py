from dataclasses import dataclass


def bonferroni_correction(p_value: float, n_tests: int) -> float:
    """
    Implement the Bonferroni correction for multiple hypothesis testing.

    Args:
        p_value: The p-value of the test.
        n_tests: The number of tests performed.

    Returns:
        The corrected p-value.
    """
    return min(1, n_tests * p_value)

def debug_fwer_function(x, y):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    breakpoint()
    return True


class GroupConfidenceFunction(dataclass):
    bonferroni_correction: callable = bonferroni_correction
    debug: callable = debug_fwer_function

    @property
    def list(self):
        return list(self.__dict__.keys())