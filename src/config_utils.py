"""
Utility functions for loading parameters and initializing output folders.

This module centralizes the configuration logic used by the main program:
- Loading parameter files
- Switching between manual and file-based configuration
- Preparing output directories for saving results
"""

import os
import shutil
from Parameters import Parameters


def load_parameters(use_file=1, filename=None):
    r"""
    Load optimization parameters from either a data file or manual settings.

    This function centralizes the initialization of the :class:`Parameters` object.
    It allows switching between:

    - *Manual mode* (debug): parameters are hard-coded using
      :meth:`Parameters.set__paramManually`
    - *File mode* (recommended): parameters are loaded from a data folder
      using :meth:`Parameters.set__paramFolder`.

    Parameters
    ----------
    use_file : int, optional
        Mode selection flag:
        - ``1`` : load parameters from a file (default)
        - ``0`` : use manually defined parameters

    filename : str, optional
        Path to the parameter file used when ``use_file = 1``.
        Example:
        ``"parameters/param_VonMises.txt"``

    Returns
    -------
    Parameters
        A fully initialized Parameters instance.

    Raises
    ------
    ValueError
        If ``use_file = 1`` but ``filename`` is not provided.
    """
    parameters = Parameters()

    if use_file == 0:
        parameters.set__paramManually()
        return parameters

    if filename is None:
        raise ValueError("You must provide a parameter filename when use_file=1.")

    parameters.set__paramFolder(filename)
    return parameters



def init_output_folders(rank):
    r"""
    Create and initialize output folders for the optimization run.

    Only MPI rank 0 should execute this function to avoid race conditions.

    Parameters
    ----------
    rank : int
        MPI rank of the current process. Only rank 0 performs folder creation.

    Notes
    -----
    This function removes the previous ``res/`` directory if it exists,
    then creates a new one and initializes several result files:
    - cost_func.txt
    - cost_compliance.txt
    - lagrangian.txt
    - constraint.txt
    - max_vm.txt
    - volume.txt
    - param_lagrangian.txt
    - vm_1_hist.txt
    - vm_final_hist.txt
    """
    if rank != 0:
        return

    # Delete existing folder if present
    shutil.rmtree("res", ignore_errors=True)
    os.mkdir("res")

    # Files to initialize
    names = [
        "cost_func", "cost_compliance", "lagrangian", "constraint",
        "max_vm", "volume", "param_lagrangian",
        "vm_1_hist", "vm_final_hist"
    ]

    # Create empty files
    for n in names:
        open(f"res/{n}.txt", "x")
