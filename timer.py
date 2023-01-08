import time
from typing import Callable


def timeit(func: Callable, log_dict: dict):
    """
    A function decorator/wrapper that measures its execution time.

    :param func: The function to run.
    :param log_dict: A dictionary in which the time for each function is logged.
    :return:
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        log_dict[func.__name__] = end - start
        return result
    return wrapper
