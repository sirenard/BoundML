import multiprocessing
import os

import numpy as np
from pyscipopt import Model


def shifted_geometric_mean(values, shift=1.0):
    values = np.array(values)
    geom_mean = np.exp(np.mean(np.log(values + shift))) - shift
    return geom_mean

def _check(path):
    try:
        # Create a dummy model just to test reading
        m = Model()
        m.setParam("display/verblevel", 0)
        with open(os.devnull, 'w') as fnull:
            os.dup2(fnull.fileno(), 2)
            m.readProblem(path)
        return True  # Success
    except:
        return False  # Caught a Python-level error
    # If SCIP crashes (SegFault), this process dies, and the parent detects it.

def check_readability(path):
    """
    Check if a path is readable.
    It creates a child process to start a new SCIP instance that will crash if path is not valid
    """

    # --- Safety Check Step ---
    # We spawn a process to test the file.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        # Run the check asynchronously
        result = pool.apply_async(_check, (path,))

        try:
            is_safe = result.get()
        except Exception:
            is_safe = False

        return is_safe

