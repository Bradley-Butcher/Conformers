from dataclasses import dataclass


def debug_func(*args, **kwargs):
    """
    A simple admission function that has a breakpoint to investigate
    the states.
    """
    breakpoint()
    return True

@dataclass
class ComponentBase:
    debug: callable = debug_func

    @property
    def list(self):
        return list(self.__dict__.keys())