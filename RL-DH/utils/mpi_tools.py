import os, subprocess, sys
import numpy as np


def mpi_fork(n, bind_to_core=False):
    """
    Dummy function for compatibility - no longer forks processes.
    
    Args:
        n (int): Number of process to split into (ignored).
        bind_to_core (bool): Bind each MPI process to a core (ignored).
    """
    # No longer using MPI, so this is a no-op
    pass


def msg(m, string=''):
    """Print a message (single process version)."""
    print(('Message from 0: %s \t ' % string) + str(m))

def proc_id():
    """Get rank of calling process (always 0 for single process)."""
    return 0

def allreduce(*args, **kwargs):
    """No-op for single process."""
    pass

def num_procs():
    """Count active processes (always 1 for single process)."""
    return 1

def broadcast(x, root=0):
    """No-op for single process."""
    pass

def mpi_op(x, op):
    """Return input unchanged for single process."""
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    return x[0] if scalar else x

def mpi_sum(x):
    """Return input unchanged for single process."""
    return mpi_op(x, None)

def mpi_avg(x):
    """Return input unchanged for single process."""
    return mpi_op(x, None)
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x (single process version).

    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std