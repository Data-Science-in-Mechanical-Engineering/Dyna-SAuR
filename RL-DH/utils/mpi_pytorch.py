import multiprocessing
import numpy as np
import os
import torch

def setup_pytorch_for_mpi():
    """
    No-op for single process. Previously limited PyTorch thread usage for MPI.
    """
    pass

def mpi_avg_grads(module):
    """No-op for single process. Previously averaged gradients across MPI processes."""
    pass

def sync_params(module):
    """No-op for single process. Previously synced parameters across MPI processes."""
    pass

