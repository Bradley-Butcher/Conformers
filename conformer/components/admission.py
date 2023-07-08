"""A script containing a few preset admission functions."""
from conformer.components.shared import ComponentBase
from random import random


def random_admission(x, y):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    return random()

class AdmissionFunction(ComponentBase):
    random: callable = random_admission
