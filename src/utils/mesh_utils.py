"""
Mesh and level-set utilities.

This module provides helpers to:
- generate or read meshes for test cases (2D / 3D / predefined shapes)
- initialize the level-set function from analytic expressions or files
- handle mesh connectivity required by Dolfinx

Notes
-----
Functions are written to be generic: adapt the paths (mesh files) to your repository.
"""

import os
from typing import Optional, Tuple

from mpi4py import MPI
from dolfinx import mesh, io
import ufl
from dolfinx import fem


# Import or define your analytic level-set expressions here.
# The following placeholders assume you have functions defined elsewhere:
# - level_set_L_shape
# - level_set
# - level_set_3D
# If you don't, replace with your own expressions.
try:
    from levelset_expressions import level_set_L_shape, level_set, level_set_3D
except Exception:
    # Fallback placeholders; user should replace with real definitions
    def level_set_L_shape(x):
        return 0.5 - x[0]

    def level_set(x, parameters):
        return x[0] - 0.5

    def level_set_3D(x, parameters):
        return x[0] - 0.5

from dolfinx import io, mesh
from mpi4py import MPI
import create_mesh

def load_mesh(test_case, parameters, mesh_folder="mesh"):
    """
    Load or create a mesh based on the test case.
    
    :param str test_case: The test case name ("rectangle", "L_shape", or "3D").
    :param Parameters parameters: The object parameters.
    :param str mesh_folder: The folder where mesh files are stored.
    
    :returns: The mesh object.
    :rtype: dolfinx.mesh.Mesh
    """
    if test_case == "rectangle":
        # Set parameters for rectangle test case
        parameters.lx = 2
        parameters.ly = 1
        parameters.lz = 0
        # Create 2D mesh
        msh = create_mesh.create_mesh_2D(
            parameters.lx, 
            parameters.ly, 
            int(parameters.lx / parameters.h),
            int(parameters.ly / parameters.h)
        )
    
    elif test_case == "L_shape":
        # Set parameters for L-shape test case
        parameters.lx = 1
        parameters.ly = 1
        parameters.lz = 0
        # Read mesh from gmsh file
        with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
            msh, ct, _ = io.gmshio.read_from_msh(
                f"{mesh_folder}/rectangle.msh", 
                MPI.COMM_WORLD, 
                0, 
                gdim=2
            )
            xdmf.write_mesh(msh)
    
    elif test_case == "3D":
        # Set parameters for 3D test case
        parameters.lx = 2
        parameters.ly = 1
        parameters.lz = 1
        # Create 3D mesh
        msh = create_mesh.create_mesh_3D(
            parameters.lx,
            parameters.ly,
            parameters.lz,
            int(parameters.lx / parameters.h),
            int(parameters.ly / parameters.h),
            int(parameters.lz / parameters.h)
        )
    
    else:
        raise ValueError(f"Test case '{test_case}' not implemented")
    
    # Create connectivity for the mesh
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    
    return msh

def init_level_set(msh: mesh.Mesh, parameters, test_case: str) -> fem.Function:
    r"""
    Initialize the level-set function defined on the mesh.

    The function returns a :class:`dolfinx.fem.Function` defined in a standard
    Lagrange space (degree 1). The level-set expression is interpolated onto
    the function space.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        Mesh where the level-set is defined.
    parameters : Parameters
        Parameters object used by analytic expressions (if required).
    test_case : str
        Identifier of the test case to choose the analytic level-set.

    Returns
    -------
    dolfinx.fem.Function
        The level-set function interpolated on V_ls.
    """
    V_ls = fem.FunctionSpace(msh, ("Lagrange", 1))

    x = ufl.SpatialCoordinate(msh)

    if test_case == "L_shape":
        expr_ufl = level_set_L_shape(x)
    elif test_case == "rectangle":
        expr_ufl = level_set(x, parameters)
    elif test_case == "3D":
        expr_ufl = level_set_3D(x, parameters)
    else:
        raise RuntimeError(f"Test case '{test_case}' not implemented for level set initialization.")

    expr = fem.Expression(expr_ufl, V_ls.element.interpolation_points())
    ls_func = fem.Function(V_ls)
    ls_func.interpolate(expr)
    return ls_func
