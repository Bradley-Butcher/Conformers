"""A script containing a few preset group confidence functions."""

from conformer.components.shared import ComponentBase

from random import random

def random_gc(x, y):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    return random()

class GroupConfidenceFunction(ComponentBase):
    random: callable = random_gc