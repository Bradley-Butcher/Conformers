"""A script containing a few preset group confidence functions."""

from dataclasses import dataclass

def debug_gf_function(x, y):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    breakpoint()
    return True


class GroupConfidenceFunction(dataclass):
    debug: callable = debug_gf_function

    @property
    def list(self):
        return list(self.__dict__.keys())