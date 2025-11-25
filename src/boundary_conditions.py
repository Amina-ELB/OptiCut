import numpy as np
from dolfinx import fem, mesh
from dolfinx.mesh import meshtags
from petsc4py.PETSc import ScalarType
import ufl
def clamped_boundary_cantilever(x):
    return np.isclose(x[0], 0)

def clamped_boundary_L_shape(x):
    return (x[1] > (1. - 1e-6))

def define_boundary_conditions(test_case, msh, V):
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
        raise ValueError("Test case not implemented")

    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    return bc

def load_marker(test_case, parameters):
    def marker(x):
        if test_case == "L_shape":
            return np.logical_and(x[0] > (parameters.lx - 1e-6), x[1] > 0.35)
        elif test_case == "3D":
            R = 0.15
            return np.logical_and((x[0] >= parameters.lx - 1e-6), ((x[2] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - R ** 2) < 0)
        elif test_case == "rectangle":
            return np.logical_and(np.isclose(x[0], parameters.lx), np.logical_and(x[1] < (0.55), x[1] > (0.45)))
        else:
            raise ValueError("Test case not implemented")
    return marker

import numpy as np

def initialize_boundary_conditions(test_case, msh,V, V_velocity, parameters):
    """Initialize boundary conditions for the given test case."""
    dim = msh.topology.dim
    fdim = dim - 1  # facet dimension

    # Define Dirichlet boundary conditions
    bc = define_boundary_conditions(test_case, msh, V)
    bcs = [bc]

    # Neumann condition initialization for load traction
    load_marker_func = load_marker(test_case, parameters)
    facet_indices, facet_markers = [], []
    facets = mesh.locate_entities(msh, fdim, load_marker_func)

    boundary_dofs = fem.locate_dofs_geometrical(V_velocity, load_marker_func)  # collect dofs for Dirichlet bc

    bc_velocity = fem.dirichletbc(ScalarType(0.), boundary_dofs, V_velocity)

    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, 2))

    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    return bcs, bc_velocity, ds


def initialize_shift(test_case, msh, parameters):
    """Initialize the shift vector based on test case."""
    if test_case == "3D":
        shift = fem.Constant(msh, ScalarType((0., -parameters.strenght, 0.)))
    elif test_case == "rectangle" or test_case == "L_shape":
        shift = fem.Constant(msh, ScalarType((0., -parameters.strenght)))
    else:
        raise ValueError(f"Test case '{test_case}' not implemented")
    return shift