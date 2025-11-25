"""
Boundary conditions and shift utilities.

This module provides helpers to initialize:
- Dirichlet and Neumann boundary conditions
- Shift vectors used in CutFEM formulations
"""
"""
Boundary conditions and shift utilities.

Provides helpers for:
- Dirichlet and Neumann boundary conditions
- Shift vectors for CutFEM formulations
"""

import numpy as np
from dolfinx import fem, mesh
from dolfinx.mesh import meshtags
from petsc4py.PETSc import ScalarType
import ufl

def clamped_boundary_cantilever(x):
    """Return True for clamped boundary in cantilever cases."""
    return np.isclose(x[0], 0)

def clamped_boundary_L_shape(x):
    """Return True for clamped boundary in L-shape."""
    return (x[1] > (1. - 1e-6))

def define_boundary_conditions(test_case, msh, V):
    """
    Define Dirichlet BCs for a given test case.

    Parameters
    ----------
    test_case : str
        Name of the test case ("rectangle", "L_shape", "3D").
    msh : dolfinx.mesh.Mesh
        The mesh.
    V : dolfinx.fem.FunctionSpace
        Displacement space.

    Returns
    -------
    bc : dolfinx.fem.DirichletBC
        Dirichlet boundary condition for displacement.
    """
    dim = msh.topology.dim
    fdim = dim - 1  # facet dimension

    if test_case == "L_shape":
        boundary_facets = mesh.locate_entities_boundary(msh, fdim, clamped_boundary_L_shape)
        u_D = np.array([0, 0], dtype=ScalarType)
    elif test_case == "3D":
        boundary_facets = mesh.locate_entities_boundary(msh, fdim, clamped_boundary_cantilever)
        u_D = np.array([0, 0, 0], dtype=ScalarType)
    elif test_case == "rectangle":
        boundary_facets = mesh.locate_entities_boundary(msh, fdim, clamped_boundary_cantilever)
        u_D = np.array([0, 0], dtype=ScalarType)
    else:
        raise ValueError(f"Test case {test_case} not implemented")

    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    return bc

def load_marker(test_case, parameters):
    """
    Returns a function to mark facets for Neumann BCs (traction/load).

    Parameters
    ----------
    test_case : str
        Name of the test case.
    parameters : Parameters
        Parameters object with geometry info.

    Returns
    -------
    marker : function
        Function returning a boolean mask of marked facets.
    """
    def marker(x):
        if test_case == "L_shape":
            return np.logical_and(x[0] > (parameters.lx - 1e-6), x[1] > 0.35)
        elif test_case == "3D":
            R = 0.15
            return np.logical_and((x[0] >= parameters.lx - 1e-6), ((x[2] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - R ** 2) < 0)
        elif test_case == "rectangle":
            return np.logical_and(np.isclose(x[0], parameters.lx), np.logical_and(x[1] < (0.55), x[1] > (0.45)))
        else:
            raise ValueError(f"Test case {test_case} not implemented")
    return marker

def initialize_boundary_conditions(test_case, msh, V, V_velocity, parameters):
    """
    Initialize Dirichlet and Neumann boundary conditions for the given test case.

    Parameters
    ----------
    test_case : str
        Name of the test case.
    msh : dolfinx.mesh.Mesh
        Mesh object.
    V : dolfinx.fem.FunctionSpace
        Displacement space.
    V_velocity : dolfinx.fem.FunctionSpace
        Velocity space.
    parameters : Parameters
        Parameters object containing BC info.

    Returns
    -------
    bcs : list of fem.DirichletBC
        List of Dirichlet boundary conditions.
    bc_velocity : fem.DirichletBC
        Velocity BC.
    ds : ufl.Measure
        Facet measure for Neumann BCs.
    """
    bc = define_boundary_conditions(test_case, msh, V)
    bcs = [bc]

    # Neumann condition
    load_marker_func = load_marker(test_case, parameters)
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities(msh, fdim, load_marker_func)
    boundary_dofs = fem.locate_dofs_geometrical(V_velocity, load_marker_func)
    bc_velocity = fem.dirichletbc(ScalarType(0.), boundary_dofs, V_velocity)

    facet_markers = np.full_like(facets, 2)
    facet_tag = meshtags(msh, fdim, facets, facet_markers)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    return bcs, bc_velocity, ds

def initialize_shift(test_case, msh, parameters):
    """
    Initialize the shift vector for CutFEM linear form.

    Parameters
    ----------
    test_case : str
        Test case name.
    msh : dolfinx.mesh.Mesh
        Mesh object.
    parameters : Parameters
        Parameters object.

    Returns
    -------
    shift : dolfinx.fem.Function or Constant
        Shift vector.
    """
    if test_case == "3D":
        shift = fem.Constant(msh, ScalarType((0., -parameters.strenght, 0.)))
    elif test_case in ["rectangle", "L_shape"]:
        shift = fem.Constant(msh, ScalarType((0., -parameters.strenght)))
    else:
        raise ValueError(f"Test case '{test_case}' not implemented")
    return shift
