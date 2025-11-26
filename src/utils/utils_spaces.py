import ufl
from dolfinx import mesh, io, fem
from petsc4py import PETSc
import numpy as np
import dolfinx.mesh
from mpi4py import MPI



def init_function_spaces(msh):
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
    V_vm = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim, )))
    V_ls = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(msh, ("DG", 0))
    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))

    return {"V": V, "V_vm": V_vm, "V_ls": V_ls, "Q": Q, "V_DG": V_DG}
