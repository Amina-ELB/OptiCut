"""
Function spaces utilities.

Provides helpers to:
- initialize the finite element function spaces used throughout the code
- optionally build frequently used Trial/Test functions and solution Functions

Keep this module small: it only defines spaces and lightweight constructors.
"""

from typing import Dict, Tuple

from dolfinx import fem
import ufl
from mpi4py import MPI


def init_function_spaces(msh) -> Dict[str, fem.FunctionSpace]:
    r"""
    Create and return the finite element spaces used by the solver.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        Mesh object where the spaces will be defined.

    Returns
    -------
    dict
        A dictionary containing:
        - "V"    : vector Lagrange space for displacement (degree 1)
        - "V_vm" : Lagrange space for Von Mises post-processing (degree 2)
        - "V_ls" : scalar Lagrange space for the level-set (degree 1)
        - "Q"    : DG0 scalar space
        - "V_DG" : DG0 vector space (for cell-wise fields such as normals)
    """
    # displacement: vector Lagrange P1
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    # von Mises / post-processing: higher order to get smoother stresses
    V_vm = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim, )))

    # level-set: scalar P1
    V_ls = fem.functionspace(msh, ("Lagrange", 1))

    # scalar DG0
    Q = fem.functionspace(msh, ("DG", 0))

    # vector DG0 (cell-wise vector fields)
    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))

    return {"V": V, "V_vm": V_vm, "V_ls": V_ls, "Q": Q, "V_DG": V_DG}


def build_trial_test_and_functions(spaces: Dict[str, fem.FunctionSpace]) -> Dict[str, object]:
    r"""
    Convenience constructor that builds commonly used Trial/Test functions and solution Functions.

    This helper is not mandatory but keeps main.py compact: call it after
    `init_function_spaces` to obtain `u`, `v`, `uh`, `ph`, and level-set related objects.

    Parameters
    ----------
    spaces : dict
        Dictionary returned by :func:`init_function_spaces`.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - 'u'   : ufl.TrialFunction on V
        - 'v'   : ufl.TestFunction on V
        - 'uh'  : fem.Function on V (primal solution)
        - 'ph'  : fem.Function on V (adjoint/dual solution)
        - 'u_r' : ufl.TrialFunction on V_ls
        - 'v_r' : ufl.TestFunction on V_ls
        - 'ls_func_n' : ufl.TrialFunction on V_ls
        - 'ls_func_test' : ufl.TestFunction on V_ls

    Notes
    -----
    These objects are lightweight wrappers around the spaces and do not allocate heavy PETSc
    structures. They are safe to create once in the main script.
    """
    V = spaces["V"]
    V_ls = spaces["V_ls"]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    uh = fem.Function(V)
    ph = fem.Function(V)

    u_r = ufl.TrialFunction(V_ls)
    v_r = ufl.TestFunction(V_ls)
    ls_func_n = ufl.TrialFunction(V_ls)
    ls_func_test = ufl.TestFunction(V_ls)

    return {
        "u": u,
        "v": v,
        "uh": uh,
        "ph": ph,
        "u_r": u_r,
        "v_r": v_r,
        "ls_func_n": ls_func_n,
        "ls_func_test": ls_func_test,
    }
