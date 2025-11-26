"""
Output utilities for CutFEM topology optimization.

Provides functions to:
- initialize output folders
- create empty result files
"""

import os
import shutil
from mpi4py import MPI

def init_output_folders(rank: int):
    """
    Initialize the output folder and empty result files.

    Only the rank 0 process creates the folders/files in parallel runs.

    Parameters
    ----------
    rank : int
        MPI rank of the current process.
    """
    if rank != 0:
        return

    # Remove previous results folder if it exists
    shutil.rmtree("res", ignore_errors=True)
    os.mkdir("res")

    # Names of output files
    file_names = [
        "cost_func",
        "cost_compliance",
        "lagrangian",
        "constraint",
        "max_vm",
        "volume",
        "param_lagrangian",
        "vm_1_hist",
        "vm_final_hist"
    ]

    # Create empty files
    for name in file_names:
        open(f"res/{name}.txt", "x")

def write_to_file(filename: str, text: str, mode: str = "a"):
    """
    Write text to a file in the 'res' folder.

    Parameters
    ----------
    filename : str
        Name of the file (without 'res/' prefix).
    text : str
        Text to write.
    mode : str, optional
        File mode: 'w' to overwrite, 'a' to append. Default is 'a'.
    """
    filepath = os.path.join("res", filename)
    with open(filepath, mode) as f:
        f.write(text + "\n")
