from conformer.components.shared import ComponentBase

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

def none_debug(p_value: float, n_tests: int) -> float:
    """
    A debug function that does noth
    """
    return p_value

class FWERFunction(ComponentBase):
    bonferroni_correction: callable = bonferroni_correction
    none_debug: callable = none_debug 